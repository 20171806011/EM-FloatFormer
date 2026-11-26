# Full Verbatim Multi-Round Conversation — Track `M4`

- **CLI Runtime**: Codex CLI (`codex exec`)
- **Session / Thread ID**: `01a21948-980b-7413-a967-3074c32be83a`
- **Round 1 Events**: `132` raw JSONL events (`round1_events.jsonl`)
- **Round 2 Events**: `146` raw JSONL events (`round2_events.jsonl`)
- **Completeness Guarantee**: All user prompts (`prompt_r1.md`, `prompt_r2.md`), all completed reasoning/tool/command events, and final responses (`reply_r1.md`, `reply_r2.md`) are included verbatim below without truncation or paraphrasing.

---

## Round 1 — Initial Mathematical Brief (`prompt_r1.md`)

````markdown
# Track M4 (Round 1): Real-Data Distribution-Shift Bounds, Pluto/CTTC Benchmark Adaptation & Modal Free-Tier Architecture

You are **Track M4**, part of a 4-track mathematical & architectural multi-agent collaboration preparing **EM-FloatFormer Stage 4** (`docs/LONG_TERM_RESEARCH_PLAN.md` Stage 4: real SDR data validation on ADALM-PLUTO 850 MHz + CTTC 433/708/2450 MHz paired with frozen Qwen3.5-4B / fixed encoder alignment).

## 0. Strict Workspace, Environment, and Immutability Rules
- **Repository Root**: `C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4`
- **Your Writable Directory**: `research/stage4_math_and_design/M4/` ONLY.
- **FORBIDDEN Edits**: Do NOT modify or delete any files outside `research/stage4_math_and_design/M4/`. (To inspect real-data code on remote git branches, use read-only `git show origin/research/...` — do NOT check out other branches or alter git HEAD!)
- **Python Environment Guardrail**: System/Anaconda `python` crashes with MKL `libiomp5md.dll` conflicts on this machine. **ALWAYS** execute Python scripts via:
  ```powershell
  uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B <script.py>
  ```

## 1. Grounded Context to Inspect First (Read-Only)
1. `docs/LONG_TERM_RESEARCH_PLAN.md` (especially Stage 4 specification & Phase goals)
2. `research/matched_cache/learned.py` (`equivariant_b8_s17` input assumption: 3 complex samples `(3,)` at unit-energy QPSK + AWGN SNR=2 dB)
3. `research/stage3d_qwen_signed_phase/PROTOCOL_LOCK.json` and `REPORT.md` (budget note: `$17.12` spent on Stage 3D out of `$30.00` Modal free credit, **`~$12.88` remaining free compute credits**, `Billed Cost: $0.00` hard constraint!)
4. Read-only `git show` on existing real-data branches in this repo:
   - `git show origin/research/realdata-pluto-qpsk:research/realdata_pluto_qpsk/PROTOCOL.md`
   - `git show origin/research/realdata-pluto-qpsk:research/realdata_pluto_qpsk/prepare_benchmarks.py`
   - `git show origin/research/realdata-pluto-qpsk:research/realdata_pluto_qpsk/waveform_frontend.py`
   - `git show origin/research/realdata-pluto-qpsk:research/realdata_pluto_qpsk/receivers.py`
   - `git show origin/research/realdata-pluto-qpsk:research/realdata_pluto_qpsk/download_data.py`
   - `git show origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/PROTOCOL.md`
   - `git show origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/adapter.py`
   - `git show origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/experiments.py`
   - `git show origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/download_cttc.py`

### Summary of Existing Real-Data Physics in This Repo
- **ADALM-PLUTO (`QPSKlink.mat`, Mendeley `10.17632/k72b4pnc5g.1`, `822.76 MB`)**:
  - `7,096` frames × `7,560` raw IQ samples (`SPS = 20`, 126 QPSK symbols × 3 copies; Sync `[0:13]`, Online Pilot `[13:21]`, Holdout Probe `[21:26]`, Payload `[26:126]`).
  - Hardware AC-coupling / zero-IF DC cancellation $y_t[k] = h_t(s_t[k] - \bar{s}_t) + \text{ISI} + w_t[k]$ ($r = 0.999$), fading $|h_t| \in [0.0091, 0.0308]$ (`>10.6 dB`), 20-frame hardware batch jumps (p90 `36.4%`, max `0.94 rad`), `76.4%` remote recurrence, plus 8 real/stress regimes (`raw_original`, `awgn_snr{10,6,2,-2}db`, `cfo_0p015_raw`, `cfo_0p020_snr6db`, `phase_jump_snr6db`). Zero-leakage sequential splits: `train [200,3200)`, `dev [3500,4500)`, `test [4800,7040)` (`G = 16` clusters × 140 frames), independent warmup pools (`200` frames each) and `100`-frame guard bands.
- **CTTC USRP X310 (`10.5281/zenodo.4895133`, 9 captures `3 bands × 3 SNRs`, `~1.9 GB`)**:
  - `256`-symbol packets (`SPS = 4`, RRC match-filtered `file_2.dat`; `KNOWN_IDX` = 10 symbols, `HOLDOUT_IDX` = 16 symbols, `PAYLOAD_IDX` = 230 symbols).
  - Real crystal SFO (`+0.39..+0.55 ppm`), large carrier CFO (`-1285..-158 Hz`), `-3.22 dB` (`f708_snr0`) to `+24.01 dB` (`f708_snr20`), hardware coherent spur (`0.260 @ +1503 Hz` on `f708_snr10`), EVEN/ODD bit offset (`f433_snr20/snr0` ODD=1).
  - Zero-leakage cyclic source isolation (`>= 4,096`-symbol guards, `G = 72` session-loop clusters; `train=1440, dev=864, test=1440` + `1080` stress packets).

## 2. Mathematical & Architectural Questions You Must Answer Rigorously
1. **Distribution-Shift & Misspecification Bounds for Frozen `equivariant_b8_s17` on Real Hardware Channels**:
   `equivariant_b8_s17` (`learned.py`) takes 3 complex samples per symbol window `(y[0], y[1], y[2])` calibrated for unit-amplitude `QPSK/sqrt(3) + CN(0, sigma_2dB^2)` (`sigma_2dB^2 = 1 / 10^0.2 = 0.63096`). On Pluto & CTTC, matched-filtered symbol observations are scalar complex symbols $\tilde{y}_k = h_k s_k + d_k + w_k$ at arbitrary amplitude $|h_k| \in [0.009, 0.031]$ (or arbitrary USRP gain) and arbitrary SNR (`-3.2 dB` to `+24 dB`), OR windowed groups of symbols!
   - **Precisely compare the two candidate Real-Data $\to$ `s17` Symbol-Window Representations**:
     - *Representation R1 (Sub-symbol oversampled 3-tap slice within 1 symbol)* vs *Representation R2 (3x repeated/interpolated whitened symbol embedding $\tilde{y}_k / (\hat{a}_k \sqrt{3})$ or matched-energy projection onto `matched_sum`)* vs *Representation R3 (Causal-normalized whitened IQ mapping + Student-$t$ variance bound)*.
     - **Derive exact KL-divergence / mutual-information loss bounds** as a function of:
       1. Residual channel-gain / AGC estimation error $\hat{a} = |h|(1+\delta)$: prove the exact marginal distribution induced by dividing Complex Gaussian noise by estimated Rayleigh/Student-$t$ scale (quantifying heavy-tail excess KL $\text{KL}(p_{\text{misspec}} \| p_{\text{Bayes}})$ and proving when scale mismatch collapses or saturates `equivariant_b8_s17`'s fixed $\tanh$ MLP!).
       2. Uncompensated hardware DC offset residual $\Delta d$ & ISI correlation $\rho_{\text{ISI}}$.
       3. Residual CFO estimation error $\epsilon_\omega$ over lag $(h+\ell)$ & spurious sideband interference ($r_{\text{spur}} = 0.26$ as in `f708_snr10`).
2. **Mathematically Rigorous Stage 4 Benchmark Protocol (Zero-Leakage Bridge Between Stage 3D and Pluto / CTTC)**:
   - How should Stage 4 evaluate the **frozen `equivariant_b8_s17` encoder + Stage 4 equivariant/evidence-coupled frozen Qwen3.5-4B connector** on REAL Pluto (`2,240` test frames, `G=16`) and REAL CTTC (`1,440` test packets, `G=72`)?
   - Explicitly resolve:
     - **Single-symbol Q1/Q2/Q3 pairs from REAL hardware sequences**: how to pair real consecutive/separated payload/pilot windows `(k_1, k_2, k_3)` with real revealed causal pilot references (STRICTLY from `ONLINE_PILOT` / `KNOWN_IDX` / causal warmup history, NEVER touching `HOLDOUT` or future frames!) so that every prompt tests real hardware fading, real DC/ISI, real CFO/phase jumps, and real high/low SNR — AND connect that cleanly to downstream **symbol SER / EVM / bit NLL**?
     - **Exact Causality & Zero-Leakage Audit**: prove zero overlap with `HOLDOUT` probes, zero cross-split source/frame leakage, exact separation of residual CFO observable ONLY from causal sync/pilots (never oracle!).
     - **Ablation Design on Real Data**: what exact baselines must run on the IDENTICAL real-data test tuples? (Physical Bayes / `B3_mixture` ceiling vs Direct `s17` table vs Frozen LLM `Zero`-arm vs `Naive-Additive` vs `Stage4-Equivariant-Coupled` vs `Stage4+RefSufficientPool`!).
3. **Exact Modal Free-Credit Execution & Resource Plan (`<= $10.00` target inside `~$12.88` remaining credits)**:
   - Inspect exact dataset sizes, HTTP APIs (`data.mendeley.com/public-api/...`, `zenodo.org/api/...`), and runtime numbers from Stage 3D's `PROTOCOL_LOCK.json` / `REPORT.md`.
   - Design the exact Modal execution topology (dataset cache Volume -> CPU feature/frontend preprocessing -> A100/L4/A10G batched KV-cached connector training & evaluation) with **explicit arithmetic bounds** on GPU hours, Modal $/hr rates, VRAM, and total Modal credit consumption so **both** Stage 4 Synthetic Extrapolation (`|\theta| > \pi/2`, Q2 `RefPool`) **AND** Real Pluto + Real CTTC complete cleanly well under `$12.88`!
4. **Mandatory CPU Empirical Verification (`verify_m4.py`)**:
   - Write `research/stage4_math_and_design/M4/verify_m4.py` using `uv run --python 3.12 ...`:
     - Verify the exact KL misspecification & Student-$t$ / gain-error / spur / CFO-error capacity curves numerically against `equivariant_b8_s17.npz` (loaded from `research/matched_cache/weights/equivariant_b8_s17.npz`!).
     - Empirically verify on `equivariant_b8_s17` how Representation R1 vs R2 vs whitened-SNR-calibrated mapping behaves across SNR `-2..+20 dB`, AGC error `5%..30%`, residual DC `0..0.3`, and CTTC-like `0.26` coherent spur!
     - Verify the exact split index invariants for Pluto & CTTC benchmarks and verify the Modal budget table arithmetic in `results_m4.json`.
     - Optionally check (without failing if offline) HTTP reachability/metadata parity of Mendeley/Zenodo endpoints or explicitly verify downloaded manifest URLs/SHA256 strings in `results_m4.json`.

## 3. Deliverables Required from Track M4
1. `research/stage4_math_and_design/M4/THEOREM_M4.md`
2. `research/stage4_math_and_design/M4/verify_m4.py` (executed cleanly via `uv run`)
3. `research/stage4_math_and_design/M4/results_m4.json`
4. Crisp executive summary in your final message (`reply_r1.md`) reporting exact misspecification bounds, recommended `s17` real-data mapping & zero-leakage tuple protocol, exact Modal budget math, and interface questions for M1/M2/M3 Round 2.
````


## Round 1 — Execution & Reasoning Trace (`round1_events.jsonl`)

### Event #1: `thread.started` (thread_id=`01a21948-980b-7413-a967-3074c32be83a`)

### Event #2: `turn.started`

### Event #3: `item.completed` — `agent_message` (`item_0`)

I’ll read the Stage 4 plan, frozen encoder, Stage 3D billing records, and real-data branches first. Then I’ll write the bounds and benchmark protocol, and run the CPU verification with the specified `uv` environment.

### Event #4: `item.completed` — `command_execution` (`item_2`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "git status --short; git branch -r; rg --files -g AGENTS.md -g '*LONG_TERM*' -g '*PROTOCOL_LOCK*' -g '*equivariant_b8_s17*' -g '*REPORT.md' -g 'learned.py'"
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
  origin/HEAD -> origin/main
  origin/main
  origin/research/abcde-convergence-2026-11-22
  origin/research/fourphase-matched-cache
  origin/research/fourphase-memory-pilot
  origin/research/fourphase-paired-questions
  origin/research/p10-payload-channel-reconstruction
  origin/research/p11-wavefront-payload-assimilation
  origin/research/p12-first-principles-diagnostics
  origin/research/p12-stage5-offgrid-neural
  origin/research/p6-math-followup-2026-11-19
  origin/research/p6-observation-consistency-2026-11-19
  origin/research/p6-soft-information-2026-11-19
  origin/research/p7-multihypothesis-routing
  origin/research/p8-coherent-transport-manifold
  origin/research/p8-continuous-ofdm-tracking
  origin/research/p8c-ofdm-single-carrier
  origin/research/p9-cross-waveform-trust-gating
  origin/research/payload-reconstruction-manuscript
  origin/research/realdata-cttc-indoor-qpsk
  origin/research/realdata-ofdm-theory-and-stress
  origin/research/realdata-pluto-qpsk
  origin/research/realdata-sdr-ofdm
  origin/research/stage3-third-window-ministral
  origin/research/stage3b-math-optimal-readout
  origin/research/stage3c-math-before-hosts
  origin/research/stage3c-work-a
  origin/research/stage3c-work-b
  origin/research/stage3c-work-c
  origin/research/stage3c-work-d
  origin/research/stage3c-work-e
  origin/research/stage3c-work-f
  origin/research/stage3c-work-g
  origin/research/stage3d-qwen-signed-phase
P2D_REPORT.md
P2C_REPORT.md
P2B_REPORT.md
P2A_REPORT.md
P1E_REPORT.md
P1D_REPORT.md
P1C_REPORT.md
P1B_REPORT.md
P3B_REPORT.md
P2_SUPPLEMENT_REPORT.md
p12\P12_REPORT.md
P2E_REPORT.md
p11\P11_REPORT.md
research\stage3_ministral_cache\results\direct\pre_reveal_jsonl\C6_decoherent_order__equivariant_b8_s17.jsonl
research\stage3_ministral_cache\results\direct\pre_reveal_jsonl\C5_snr_6db__equivariant_b8_s17.jsonl
research\stage3_ministral_cache\results\direct\pre_reveal_jsonl\C5_snr_0db__equivariant_b8_s17.jsonl
research\stage3_ministral_cache\results\direct\pre_reveal_jsonl\C4_cfo_ramp__equivariant_b8_s17.jsonl
research\stage3_ministral_cache\results\direct\pre_reveal_jsonl\C3_mid_pilot_cont__equivariant_b8_s17.jsonl
research\stage3_ministral_cache\results\direct\pre_reveal_jsonl\C2_discrete_walk__equivariant_b8_s17.jsonl
research\stage3_ministral_cache\results\direct\pre_reveal_jsonl\C1_coherent_2db__equivariant_b8_s17.jsonl
research\stage3_ministral_cache\REPORT.md
research\stage3_ministral_cache\connector_weights\connector_equivariant_b8_s17.pt
research\stage3d_qwen_signed_phase\REPORT.md
research\stage3d_qwen_signed_phase\PROTOCOL_LOCK.json
research\stage3c\work_results\E\REPORT.md
research\stage3c\work_results\D\REPORT.md
research\stage3c\work_results\C\REPORT.md
research\stage3c\work_results\B\REPORT.md
research\stage3c\E\REPORT.md
research\realdata_ofdm_stress_and_theory\REPORT.md
research\matched_cache\weights\equivariant_b8_s17.npz
research\matched_cache\results\test\equivariant_b8_s17.json
research\matched_cache\results\dev\equivariant_b8_s17.json
research\matched_cache\learned.py
research\content_addressing_scalar\REPORT.md
research\abcde_convergence_2026_11_22\experiments\whole_frame_gate\REPORT.md
research\abcde_convergence_2026_11_22\experiments\parity\REPORT.md
research\abcde_convergence_2026_11_22\experiments\cfo\REPORT.md
research\abcde_convergence_2026_11_22\C\REPORT.md
research\abcde_convergence_2026_11_22\B\REPORT.md
research\abcde_convergence_2026_11_22\A\REPORT.md
p9\P9_REPORT.md
p8c\P8C_REPORT.md
p8\P8_REPORT.md
p7\P7_REPORT.md
p6_softinfo\REPORT.md
p6_followup\REPORT.md
p6_consistency\REPORT.md
exploratory\p3de_tdl_surrogate\P3DE_SURROGATE_REPORT.md
exploratory\16qam_joint_pa_probe\REPORT.md
docs\LONG_TERM_RESEARCH_PLAN.md
P3_A_MECHANISM_REPORT.md
P3_0_A_REPORT.md
P3C_REPORT.md
p4\P4_CONTINUATION_REPORT.md
p4\P4C_1D_REPORT.md
p6\P6_REPORT.md
p5b\P5B_REPORT.md
p5\P5_REPORT.md
```

### Event #5: `item.completed` — `command_execution` (`item_1`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "rg -n -i 'EM.FloatFormer|equivariant_b8_s17|pluto|CTTC|stage.?3d' 'C:/Users/李则徐/.codex/memories/MEMORY.md'"
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

### Event #6: `item.completed` — `command_execution` (`item_7`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n -A 12 -B 1 'P4 scientific|matched-control' 'C:/Users/李则徐/.codex/memories/MEMORY.md'"
```

**Aggregated Output:**

```text
28-
29:# Task Group: EM-FloatFormer P4 scientific assessment and reproducible publication
30-
31-scope: Assess and publish the EM-native Transformer P4 simulation without overstating learned-model performance.
32:applies_to: cwd=C:\Users\李则徐\Documents\ChatGPT\EM-FloatFormer; reuse_rule=Reuse the matched-control, reproducibility, and publication approach for this repository; numerical results, commit, hardware status, and receiver outcomes are P4-specific.
33-
34-## Task 1: Complete and assess P4 EM-native Transformer research, success
35-
36-### rollout_summary_files
37-
38-- rollout_summaries/2026-11-17T20-54-26-dFGP-p4_em_native_transformer_assessment_and_github_publication.md (cwd=\\?\C:\Users\李则徐\Documents\ChatGPT\EM-FloatFormer, rollout_path=\\?\C:\Users\李则徐\.codex\sessions\2026\11\17\rollout-2026-11-17T20-54-26-01a1eb4a-230f-7e02-b32d-21348996e25b.jsonl, updated_at=2026-11-18T16:17:48+00:00, thread_id=01a1eb4a-230f-7e02-b32d-21348996e25b, formal P4-D and assessment completed)
39-
40-### keywords
41-
42-- EM-FloatFormer, P4, EMFloatFormer2D, C*-equivariance, Hermitian attention, complex RoPE, sparse DMRS, Doppler, CFO, Rapp PA, LDPC, P4D, Holm, sign-flip
43-
44-## Task 2: Publish combined P4 research to GitHub, success
```

### Event #7: `item.completed` — `command_execution` (`item_6`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command 'foreach ($file in '"'PROTOCOL.md','adapter.py','experiments.py','download_cttc.py') { git show \"origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/"'$file" }'
```

**Aggregated Output:**

```text
# EM-FloatFormer 实测专项实验协议（CTTC 室内多频段多信噪比 QPSK）

- **项目基线提交**：`f566f9b8b0ddfb468a4cb3fd64588bfa46c87212`（分支 `research/p12-first-principles-diagnostics`）
- **本专项工作分支**：`research/realdata-cttc-indoor-qpsk`
- **本专项交付目录**：`research/realdata_cttc_qpsk/`
- **理论与源码衔接**：
  - `p12/P12_CONSISTENCY_FIX.md` 与 `p12/offgrid_receiver.py`（单次计分先验一致性、防泄漏留出验证、分折/分组隔离）
  - `research/content_addressing_scalar/REPORT.md`（固定局部窗口下界、全局内容寻址精确风险公式 (2)、范围增益 vs 内容识别方差分解公式 (7)、过冷/过热与共享观测偏差反例）
  - `research/posterior_fusion/pages/01.md–12.md`（信道后验均值正交投影定理、总协方差分支内+分支间分解、MSE 与 BER 排序反转反例、完整混合后验进入逐符号/逐比特 MAP 与 log-sum-exp LLR、平均 LLR 负对照）

---

## 1. 数据来源与完整性验证

- **官方数据集**：Adriano Pastore, Armin Ghani (Centre Tecnològic de Telecomunicacions de Catalunya, CTTC), *Datasets of Indoor Wireless Channel Measurements for Machine Learning Applications* (v1.0, ARISTIDES Project Grant RTI2018-099722-B-I00), Zenodo, June 2, 2021.
- **DOI**：`10.5281/zenodo.4895133`
- **官方链接**：`https://zenodo.org/records/4895133`
- **API 端点**：`https://zenodo.org/api/records/4895133`
- **许可协议**：CC-BY-4.0
- **纳入实测会话**：全量 **3 频段 × 3 名义 SNR = 9 个 QPSK 实测采集会话**（全部于 2021-05-19 在 CTTC 室内静态实验室由 Ettus Research USRP X310 + VERT2450 单垂直全向天线录制），以及 `README.pdf` 与 `sources.tar.gz`（含 `source_QPSK.dat`）。全部原始压缩包与解压文件的精确大小、官方 MD5 与实测 SHA-256 记录于 `data_manifest.json` 及 `download_cttc.py`。

### 1.1 波形与物理处理阶段审计
1. **发送端**：`source_QPSK.dat` 为 1,000,000 个随机字节（`uint8`，取值 `{0, 1}`），按 Gray 码映射为 500,000 个单位能量复数 QPSK 符号 $x_k = \frac{1}{\sqrt{2}}[(1-2b_{2k}) + j(1-2b_{2k+1})]$，循环重复发射（`Repeat: Yes`，周期恰为 1.0 s）。符号率 $f_{\text{symb}} = 0.5\text{ MS/s}$（1 Mbps，无信道纠错编码，故**原始编码位 BER = 信息位 BER**），TX RRC 脉冲成形滚降系数 $\alpha = 0.2$、256 taps、2× 插值（$f_{\text{sample,TX}} = 1\text{ MS/s}$）。
2. **接收端**：RX 采样率 $f_{\text{sample,RX}} = 2\text{ MS/s}$（标称过采样因子 $\text{SPS} = 4$ samples/symbol）。每个会话录制约 11.001 秒（22,001,908 个 `complex64` 样本），开头约 1 秒发射机静默用于背景噪声方差估计。
   - `file_1.dat`：ADC 下变频后原始复基带 I/Q（未过 RX RRC 匹配滤波）。
   - `file_2.dat`：经 GNU Radio 256-tap RRC 匹配滤波（Decimation = 1，仍保持 4 samples/symbol）后的复基带 I/Q。
3. **一手实测物理勘误与特性发现**：
   - **比特奇偶对齐偏移**：7/9 会话为偶数比特对齐（`bits[0::2], bits[1::2]`），但 `f433_snr20_QPSK` 与 `f433_snr0_QPSK` 存在 1 比特奇数偏移（`bits[1::2], bits[2::2]`）。`adapter.py` 通过自动互相关锁定唯一对齐方式并记录 `bit_alignment`，使全部 3 个 `snr20` 会话均达到零误码基线。
   - **晶振采样率偏差（SFO）与载波频偏（CFO）**：全会话实测存在 `+0.39 ~ +0.55 ppm` 的真实晶振时钟偏差（10 秒漂移 > 2 个符号）及对应载频比例 CFO（2450 MHz 约 `-1175 ~ -1285 Hz`，708 MHz 约 `-370 ~ -395 Hz`，433 MHz 约 `-157 ~ -218 Hz`）。
   - **真实硬件杂散/多普勒双峰**：`f708_snr10_QPSK` 实测存在幅度高达主峰 `26.0%`、频偏 `+1510.6 Hz` 的相干边带，引发 `16.5%` 幅度拍频与剧烈瞬时频偏摆动。

---

## 2. 标准化数据包切分与四重零泄漏防火墙

为消除 `source_QPSK.dat` 在 ~9 秒录制内循环重复及跨会话复用导致的潜在序列/窗口泄漏，强制执行以下不可违背规则：

1. **包结构（$L_{\text{pkt}} = 256$ 符号 = 512 比特 = 0.512 ms）**：
   - **同步头（`SYNC_IDX`）**：索引 `0..5`（6 符号，接收机可见）
   - **稀疏包内导频（`PILOT_IDX`）**：索引 `126, 127, 254, 255`（4 符号，接收机可见；与同步头合计 `KNOWN_IDX` = **10 符号**，导频开销 `3.91%`）
   - **留出验证导频（`HOLDOUT_IDX`）**：索引 `40..47` 与 `200..207`（16 符号，**严格留出，接收机不可见**，专用于独立评估留出信道估计 NMSE 与高斯预测 NLL）
   - **载荷（`PAYLOAD_IDX`）**：其余 **230 符号**（460 原始/信息比特，**接收机完全盲测**）
   - 每个包前后附带 `PAD_SYMS = 8` 符号（32 RX 样点）保护波形用于分数定时插值与多抽头均衡。
2. **源比特区间绝对互斥（含 4,096 符号保护带，覆盖 EVEN/ODD 对齐偏移）**：
   - `train` 目标包：`[0, 160_000)`；`train` 专属参考池：`[164_096, 200_000)`
   - `dev` 目标包：`[204_096, 290_000)`；`dev` 专属参考池：`[294_096, 330_000)`
   - `test` 目标包：`[334_096, 455_000)`；`test` 专属参考池：`[459_096, 495_904)`
   - 环绕保护带：`[495_904, 500_000)`（4,096 符号）。
3. **时间间隔隔离**：每个会话在稳定区间（$t \ge 1.45\text{s}$）抽取连续 8 个 1 秒循环（`loop_id = 0..7`，9 会话 × 8 循环 = **72 个独立会话-循环簇 `cluster_id`**）；同循环内相邻入选包步长 $\ge 2,048$ 符号（包间空隙 $\ge 1,792$ 符号 $> 28\times$ RRC 滤波器长度），彻底杜绝滤波拖尾与短时衰落重叠。
4. **固定测试清单规模**：
   - 原始实测：`train = 1,440` 目标包（+ 576 参考包），`dev = 864` 目标包（+ 576 参考包），`test = 1,440` 目标包（+ 576 参考包，总计 331,200 载荷符号 = 662,400 比特）。
   - 独立可控物理扰动压力测试（在原始实测 IQ 上注入 3 类扰动：`stress_low_snr`、`stress_cfo_phase_jump`、`stress_burst_impairment`）：`train = 720`、`dev = 432`、`test = 1,080` 包（每类扰动 360 包，覆盖全部 9 会话与 72 簇），结果与原始实测完全分轨保存。

---

## 3. 接收机信息可见性与四组配对对照设计

### 3.1 严格信息可见性约束
| 对象 | 接收机可见 | 仅离线评测器可见 |
| :--- | :--- | :--- |
| 目标包 | `y_iq_4x`, `x_known` (10 符号), `KNOWN_IDX`, `bg_noise_var` | `x_all_true`, `HOLDOUT_IDX` 真值, `PAYLOAD_IDX` 真值, `h_offline_ref`, `offline_cfo_hz`, `capture_id` |
| 参考池包 | 仅由参考包自身 10 个已知导频提取的 `RefEntry`（CFO/定时/功率/噪声/SNR/斜率） | 参考包载荷、留出导频与 `capture_id`（仅用于离线审计匹配权重质量） |
| 调优与选模 | 所有经典超参、温度校准、网络训练早停与种子选择严格仅用 `train` / `dev` | `test` 仅在全部冻结后评测一次 |

### 3.2 四组实验口径
1. **成熟经典接收基线**：匹配滤波（`file_2.dat`，并在 `f2450_snr20` 上与 `file_1.dat` 自建 256-tap RRC 交叉复核）+ 分数定时同步 + 导频联合 ML CFO 估计 + `classical_ls` / `classical_mmse` / `classical_mmse_slope` / `classical_mmse_3tap` + QPSK 硬/软解调。
2. **对照 A（全局参考与内容匹配）**：在完全相同的跨会话全局混合参考池（$M=36$，9 会话各 4 包随机打乱，仅前 $k=4$ 为固定局部窗口）及同会话时间参考池（$M=24$，按时间距离排序，近邻 $k=4$ 为局部窗口）上，通过相同下游先验条件接收机配对比较：`classical_mmse`（无外部参考）、`A1_local_window_k4`、`A2_global_uniform`、`A3_global_snr_weight`、`A4_global_content_attn`（精确相位边缘化复高斯导频对数似然 + dev 校准温度 $\tau_A$）、过冷硬检索 `A5_content_cold_hard` 与过热 `A6_content_hot`；同步检验 `content_addressing_scalar` 风险分解与权重距离关系。
3. **对照 B（硬选 vs 线性融合 vs 保留混合后验似然）**：共享完全相同的 $L=12$ 物理竞争候选库（5 个局部多峰 CFO + 3 个互斥全局内容参考簇 + 2 个线性斜率漂移 + 2 个分段中包跳变分支）与完全相同的单次计分边缘对数证据后验权重 $\alpha_r$，配对比较：`B1_hard_select`、`B2_linear_fusion_plugin`（含总方差 = 分支内 + 分支间）、`B2b_linear_within_only`（故意省去分支间方差，检验 §4.1 失准预言）、`B3a_sym_mixture`（逐符号混合）、`B3_mixture_likelihood`（精确包级 LOO 联合混合后验进入 QPSK MAP 与 log-sum-exp LLR，验证 §5.2）与 `B4_avg_llr_antipattern`（负对照，验证 §5.2 平均 LLR 缺陷）。
4. **对照 C（轻量级可训练 Attention / 权重预测 / 残差网络）**：固定上述全部 $L=12$ 物理候选生成与后端 `mixture_likelihood` 判决器，在单线程 PyTorch CPU（`TORCH_NUM_THREADS=1`）下使用 3 个独立随机种子 `{20261121, 20261122, 20261123}` 训练 `C1_attn_weight_net`（跨候选自注意力残差权重网络）与 `C2_attn_weight_resid_net`（权重网络 + 有界复增益/相位/对数方差残差头），以可微混合 Bit BCE + 训练留出信道 MSE 联合优化，在 `dev` 上早停选模，在 `test` 上与 dev 调优后的最强经典方法 `B3_mixture_likelihood` 严格配对比较。

---

## 4. 统一指标定义与统计推断

- **分段统计**：分别输出 `sync`（6 符号）、`pilot`（4 符号）、`holdout_pilot`（16 符号）、`payload`（230 符号 = 460 比特）的指标。
- **核心指标**：SER、符号重建 MSE（raw）、原始 EVM 与无偏 EVM（消除线性收缩偏置）、BER（明确标注无信道编码：原始位 = 信息位）、**PER / FER**（事件定义：单包 230 个载荷符号 / 460 比特中存在 $\ge 1$ 个判决错误即计为 1 次包错；分母为该集总有效评估包数）、留出导频观测预测 NMSE（`holdout_obs_nmse`）、相对离线高精度参考信道的留出 NMSE（`holdout_ref_nmse`）、留出高斯信道预测 NLL（`holdout_ch_nll_nats`）、载荷软比特 NLL（`payload_bit_nll_nats`）与 15-bin ECE（`payload_bit_ece`）。
- **配对统计推断**：以 72 个独立会话-循环簇（`cluster_id`）及 9 个采集会话（`capture_id`）为独立抽样单元，计算簇级配对改善差值（正值表示候选方法优于基线）、胜/平/负计数、10,000 次分层簇 Bootstrap 95% 置信区间，以及精确（$G \le 16$ 时 $2^G$ 态）或 50,000 次蒙特卡洛配对符号翻转 $p$ 值。
- **工程成本与鲁棒性**：统一单线程（`OMP=MKL=OPENBLAS=TORCH=1`）记录每包中位/均值/P95 推理耗时、训练时间、参数量、进程峰值 RSS 内存、成功/失败/异常数及处理覆盖率。
"""Unified CTTC Indoor QPSK real-data adapter with strict zero-leakage split & packet indexing.

Reads raw pre-RRC (`file_1.dat`) and official post-RRC (`file_2.dat`) complex64 IQ streams plus
`source_QPSK.dat` and `config.txt`, performs reproducible waveform anchoring, fractional-SFO
packet slicing, and builds strictly disjoint train/dev/test target & reference pools.

Leakage firewall:
  1. Source-symbol index intervals (mod 500,000) are strictly disjoint across train/dev/test
     and between target packets and reference-pool packets, separated by 4,096-symbol guard bands.
  2. Selected packets within each loop are spaced by stride >= 2,048 symbols (gap >= 1,792 symbols
     > 28x RRC filter length), eliminating filter-tail and short-window overlap.
  3. Holdout verification pilots (16 symbols/pkt) and payload symbols (224 symbols/pkt) are never
     exposed to receivers; only `y_iq_4x`, `x_known` (sync+pilots), and `bg_noise_var` are visible.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import signal

from .evaluator import bits_to_qpsk

# --- Fixed physical waveform constants (from CTTC README.pdf & config.txt) ---
FS_RX = 2_000_000.0
FS_SYM = 500_000.0
SPS_NOM = 4
RRC_ALPHA = 0.2
RRC_NTAPS = 256
SOURCE_N_BITS = 1_000_000
SOURCE_N_SYMS = 500_000

# --- Standardized packet structure (256 symbols = 512 bits = 0.512 ms) ---
# Compact realistic pilot budget: 6 preamble sync + 4 sparse mid/tail pilots = 10 known pilots (3.9%)
PKT_LEN = 256
PAD_SYMS = 8  # 8 symbols (=32 RX samples) padding on each side for timing & multi-tap EQ
SYNC_IDX = np.arange(0, 6, dtype=np.int64)
PILOT_IDX = np.array([126, 127, 254, 255], dtype=np.int64)
KNOWN_IDX = np.concatenate([SYNC_IDX, PILOT_IDX])  # 10 known pilot symbols spanning start/mid/end
HOLDOUT_IDX = np.concatenate([np.arange(40, 48, dtype=np.int64), np.arange(200, 208, dtype=np.int64)])
_NON_PAYLOAD = set(KNOWN_IDX.tolist()) | set(HOLDOUT_IDX.tolist())
PAYLOAD_IDX = np.array([i for i in range(PKT_LEN) if i not in _NON_PAYLOAD], dtype=np.int64)

assert len(SYNC_IDX) == 6
assert len(PILOT_IDX) == 4
assert len(KNOWN_IDX) == 10
assert len(HOLDOUT_IDX) == 16
assert len(PAYLOAD_IDX) == 230
assert len(set(KNOWN_IDX) | set(HOLDOUT_IDX) | set(PAYLOAD_IDX)) == PKT_LEN

# --- Strict disjoint source-symbol intervals (mod 500,000) with 4,096-symbol guard bands ---
SPLIT_SRC_RANGES: dict[str, dict[str, tuple[int, int]]] = {
    "train": {"target": (0, 160_000), "ref": (164_096, 200_000)},
    "dev": {"target": (204_096, 290_000), "ref": (294_096, 330_000)},
    "test": {"target": (334_096, 455_000), "ref": (459_096, 495_904)},
}

ALL_CAPTURES: tuple[tuple[str, int, int, float], ...] = (
    ("f2450_snr20_QPSK", 2450, 20, -1240.0),
    ("f2450_snr10_QPSK", 2450, 10, -1175.0),
    ("f2450_snr0_QPSK", 2450, 0, -1280.0),
    ("f708_snr20_QPSK", 708, 20, -370.0),
    ("f708_snr10_QPSK", 708, 10, -375.0),
    ("f708_snr0_QPSK", 708, 0, -370.0),
    ("f433_snr20_QPSK", 433, 20, -166.0),
    ("f433_snr10_QPSK", 433, 10, -218.0),
    ("f433_snr0_QPSK", 433, 0, -157.0),
)


def make_rrc_taps(num_taps: int = RRC_NTAPS, alpha: float = RRC_ALPHA, sps: int = SPS_NOM) -> np.ndarray:
    """Design unit-energy Root Raised Cosine (RRC) FIR filter matching GNU Radio specification."""
    t = (np.arange(num_taps, dtype=np.float64) - (num_taps - 1) / 2.0) / float(sps)
    h = np.zeros(num_taps, dtype=np.float64)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            h[i] = 1.0 - alpha + 4.0 * alpha / math.pi
        elif abs(abs(4.0 * alpha * ti) - 1.0) < 1e-12:
            h[i] = (alpha / math.sqrt(2.0)) * (
                (1.0 + 2.0 / math.pi) * math.sin(math.pi / (4.0 * alpha))
                + (1.0 - 2.0 / math.pi) * math.cos(math.pi / (4.0 * alpha))
            )
        else:
            num = math.sin(math.pi * ti * (1.0 - alpha)) + 4.0 * alpha * ti * math.cos(math.pi * ti * (1.0 + alpha))
            den = math.pi * ti * (1.0 - (4.0 * alpha * ti) ** 2)
            h[i] = num / den
    h /= math.sqrt(float(np.sum(h ** 2)))
    return h


def load_source_qpsk(sources_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load `source_QPSK.dat` and return (bits_uint8 [1M], syms_even [500k], syms_odd [500k]).

    Empirically verified on Zenodo 4895133: 7 of 9 QPSK captures use even bit pairing
    (bits[0::2], bits[1::2]), whereas `f433_snr20_QPSK` and `f433_snr0_QPSK` were captured
    with a 1-bit odd offset (bits[1::2], bits[2::2] wrapped mod 1M).
    """
    p = sources_dir / "source_QPSK.dat"
    bits = np.fromfile(p, dtype=np.uint8)
    if len(bits) != SOURCE_N_BITS:
        raise ValueError(f"Expected {SOURCE_N_BITS} bits in {p}, got {len(bits)}")
    syms_even = bits_to_qpsk(bits)
    syms_odd = bits_to_qpsk(np.roll(bits, -1))
    return bits, syms_even, syms_odd


def parse_config_txt(path: Path) -> dict[str, str]:
    cfg: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or "," not in line:
            continue
        k, v = line.split(",", 1)
        cfg[k.strip()] = v.strip()
    return cfg


@dataclass
class CaptureAnchor:
    """Offline capture-level physical characterization & continuous stream anchor."""
    capture_id: str
    freq_mhz: int
    nom_snr_db: int
    timestamp: str
    n_samples: int
    bg_noise_var: float
    sig_total_pow: float
    measured_snr_db: float
    anchor_rx_pos: float  # exact fractional RX sample index of symbol `anchor_src_sym` near t=2.0s
    anchor_src_sym: int
    sps_eff: float
    sfo_ppm: float
    offline_cfo_med_hz: float
    offline_cfo_min_hz: float
    offline_cfo_max_hz: float
    secondary_spur_ratio: float
    secondary_spur_offset_hz: float
    bit_alignment: int = 0  # 0=even (bits[0::2],bits[1::2]), 1=odd (1-bit shifted)
    lock_norm_corr_mean: float = 0.0
    stream_source: str = "file_2.dat"


@dataclass
class PacketSample:
    """Single standardized real-data QPSK evaluation packet with strict receiver/evaluator separation."""
    # Public identifiers & metadata
    unit_id: str
    capture_id: str
    freq_mhz: int
    nom_snr_db: int
    cluster_id: str
    loop_id: int
    split: str
    role: str  # "target" or "ref"
    rx_start_sample: int
    frac_rx_offset: float  # sub-sample residual inside [0, 1) from stream anchor (offline meta only)
    src_sym_start: int
    time_sec: float
    bg_noise_var: float
    # Receiver-visible waveform & known pilot inputs ONLY
    y_iq_4x: np.ndarray  # complex64, shape ((PKT_LEN + 2*PAD_SYMS) * 4,)
    x_known: np.ndarray  # complex128, shape (16,) corresponding to KNOWN_IDX
    # Evaluator-only ground truth & offline reference (NEVER passed to receivers)
    x_all_true: np.ndarray  # complex128, shape (256,)
    h_offline_ref: np.ndarray  # complex128, shape (256,)
    offline_cfo_hz: float
    offline_sig_pow: float
    perturbation_tag: str = "original"
    extra_meta: dict[str, Any] = field(default_factory=dict)


def _parabolic_peak(y_m1: float, y_0: float, y_p1: float) -> float:
    denom = y_m1 - 2.0 * y_0 + y_p1
    if abs(denom) < 1e-30:
        return 0.0
    return float(np.clip(0.5 * (y_m1 - y_p1) / denom, -0.5, 0.5))


def anchor_capture(
    data_dir: Path,
    capture_id: str,
    freq_mhz: int,
    nom_snr_db: int,
    cfo_hint_hz: float,
    sym_ref_even: np.ndarray,
    sym_ref_odd: np.ndarray | None = None,
    *,
    stream_file: str = "file_2.dat",
) -> CaptureAnchor:
    """Measure background noise, detect bit alignment (even vs odd), lock loop, and extract exact SFO."""
    cap_dir = data_dir / capture_id
    cfg = parse_config_txt(cap_dir / "config.txt")
    x = np.memmap(cap_dir / stream_file, dtype=np.complex64, mode="r")
    n_samples = len(x)

    # 1) Robust silence noise floor estimation in t in [0.10s, 0.85s] using 10ms blocks
    blk = 20_000
    sil_pwrs = np.array(
        [float(np.mean(np.abs(x[i * blk : (i + 1) * blk]) ** 2)) for i in range(10, 85)],
        dtype=np.float64,
    )
    # Use lower 60% median to stay immune to transient glitches (e.g. f2450_snr10 block 7..9)
    bg_noise_var = float(np.median(np.sort(sil_pwrs)[:45]))
    sig_pwrs = np.array(
        [float(np.mean(np.abs(x[i * blk : (i + 1) * blk]) ** 2)) for i in range(200, 950, 5)],
        dtype=np.float64,
    )
    sig_total_pow = float(np.median(sig_pwrs))
    measured_snr_db = 10.0 * math.log10(max(sig_total_pow - bg_noise_var, 1e-18) / max(bg_noise_var, 1e-18))

    # 2) Detect even (0) vs odd (1) bit alignment at t=4.0s via FFT cross-correlation
    nfft_ref = 1048576
    f_ref_even = np.fft.fft(sym_ref_even, n=nfft_ref)
    cand_refs: list[tuple[int, np.ndarray, np.ndarray]] = [(0, sym_ref_even, f_ref_even)]
    if sym_ref_odd is not None:
        cand_refs.append((1, sym_ref_odd, np.fft.fft(sym_ref_odd, n=nfft_ref)))

    L_lock = 4096
    n_idx = np.arange(L_lock, dtype=np.float64)
    s_probe = int(4.0 * FS_RX)
    best_align = 0
    best_align_v = -1.0
    best_sym_ref = sym_ref_even
    best_f_ref = f_ref_even
    for al_id, s_r, f_r in cand_refs:
        for ph in range(4):
            q = np.asarray(x[s_probe + ph : s_probe + ph + 4 * L_lock : 4], dtype=np.complex128)
            q = (q - q.mean()) * np.exp(-2j * math.pi * cfo_hint_hz * n_idx / FS_SYM)
            qn = float(np.linalg.norm(q))
            corr = np.fft.ifft(f_r * np.conj(np.fft.fft(q, n=nfft_ref)))[:SOURCE_N_SYMS]
            v = float(np.max(np.abs(corr))) / (qn * math.sqrt(L_lock) + 1e-30)
            if v > best_align_v:
                best_align_v = v
                best_align = al_id
                best_sym_ref = s_r
                best_f_ref = f_r

    # 3) Multi-checkpoint FFT lock at t = 2.0, 4.0, 6.0, 8.0, 10.0 s using locked bit alignment
    locks: list[tuple[float, float, int, float, float]] = []
    spur_ratio = 0.0
    spur_offset = 0.0

    for t_s in (2.0, 4.0, 6.0, 8.0, 10.0):
        s0 = int(t_s * FS_RX)
        if s0 + 4 * 16384 + 64 >= n_samples:
            continue
        best_v, best_ph, best_lag = -1.0, 0, 0
        for ph in range(4):
            q = np.asarray(x[s0 + ph : s0 + ph + 4 * L_lock : 4], dtype=np.complex128)
            q = (q - q.mean()) * np.exp(-2j * math.pi * cfo_hint_hz * n_idx / FS_SYM)
            qn = float(np.linalg.norm(q))
            corr = np.fft.ifft(best_f_ref * np.conj(np.fft.fft(q, n=nfft_ref)))[:SOURCE_N_SYMS]
            mag = np.abs(corr) / (qn * math.sqrt(L_lock) + 1e-30)
            lag = int(np.argmax(mag))
            if float(mag[lag]) > best_v:
                best_v, best_ph, best_lag = float(mag[lag]), ph, lag
        if best_v < 0.35:
            raise RuntimeError(f"Failed to lock {capture_id} at t={t_s}s: max norm_corr={best_v:.4f}")

        # High-resolution ML CFO over 16384 symbols
        Ls = 16384
        ql = np.asarray(x[s0 + best_ph : s0 + best_ph + 4 * Ls : 4], dtype=np.complex128)
        rl = best_sym_ref[(best_lag + np.arange(Ls)) % SOURCE_N_SYMS]
        z = ql * np.conj(rl)
        nfft_z = 262144
        sz = np.abs(np.fft.fft(z * np.hanning(Ls), n=nfft_z))
        fz = np.fft.fftfreq(nfft_z, d=1.0 / FS_SYM)
        pk = int(np.argmax(sz))
        cfo_ml = float(fz[pk])

        if abs(t_s - 4.0) < 0.1:
            sz2 = sz.copy()
            sz2[np.abs(fz - cfo_ml) < 60.0] = 0.0
            pk2 = int(np.argmax(sz2))
            spur_ratio = float(sz2[pk2] / (sz[pk] + 1e-30))
            spur_offset = float(fz[pk2] - cfo_ml)

        # Sub-sample timing refinement over +-4 RX samples
        taps = np.empty(9, dtype=np.float64)
        for k, dt in enumerate(range(-4, 5)):
            qt = np.asarray(x[s0 + best_ph + dt : s0 + best_ph + dt + 4 * L_lock : 4], dtype=np.complex128)
            qt = (qt - qt.mean()) * np.exp(-2j * math.pi * cfo_ml * n_idx / FS_SYM)
            taps[k] = abs(np.mean(qt * np.conj(rl[:L_lock])))
        pt = int(np.argmax(taps))
        fdt = _parabolic_peak(taps[pt - 1], taps[pt], taps[pt + 1]) if 0 < pt < 8 else 0.0
        abs_rx = float(s0 + best_ph + (pt - 4)) + fdt
        locks.append((t_s, abs_rx, best_lag, cfo_ml, best_v))

    # Linear fit of (actual_sym_advance -> abs_rx) across all 5 checkpoints for sub-0.01 ppm SFO precision
    sym_adv = [0.0]
    rx_pos = [locks[0][1]]
    for i in range(1, len(locks)):
        d_rx = locks[i][1] - locks[0][1]
        n_loops = round((d_rx / 4.0 - (locks[i][2] - locks[0][2])) / SOURCE_N_SYMS)
        act_syms = n_loops * SOURCE_N_SYMS + (locks[i][2] - locks[0][2])
        sym_adv.append(float(act_syms))
        rx_pos.append(locks[i][1])
    slope, intercept = np.polyfit(np.array(sym_adv), np.array(rx_pos), 1)
    sps_eff = float(slope)
    sfo_ppm = (sps_eff / 4.0 - 1.0) * 1e6
    cfos = [lk[3] for lk in locks]
    corrs = [lk[4] for lk in locks]

    return CaptureAnchor(
        capture_id=capture_id,
        freq_mhz=freq_mhz,
        nom_snr_db=nom_snr_db,
        timestamp=cfg.get("time", ""),
        n_samples=n_samples,
        bg_noise_var=bg_noise_var,
        sig_total_pow=sig_total_pow,
        measured_snr_db=measured_snr_db,
        anchor_rx_pos=float(intercept),
        anchor_src_sym=locks[0][2],
        sps_eff=sps_eff,
        sfo_ppm=sfo_ppm,
        offline_cfo_med_hz=float(np.median(cfos)),
        offline_cfo_min_hz=float(np.min(cfos)),
        offline_cfo_max_hz=float(np.max(cfos)),
        secondary_spur_ratio=spur_ratio,
        secondary_spur_offset_hz=spur_offset,
        bit_alignment=best_align,
        lock_norm_corr_mean=float(np.mean(corrs)),
        stream_source=stream_file,
    )


def _offline_reference_channel(
    y_iq_4x: np.ndarray,
    frac_offset: float,
    x_all_true: np.ndarray,
    cfo_hint_hz: float,
) -> tuple[np.ndarray, float, float]:
    """Compute high-precision offline reference channel h(n) over all 256 symbols (EVALUATOR ONLY).

    Uses cubic fractional resampling at true stream offset, ML CFO over all 256 known symbols,
    and gentle 2nd-order complex polynomial fit in time to track real intra-packet drift/beat.
    """
    from scipy.interpolate import CubicSpline

    t_rx = np.arange(len(y_iq_4x), dtype=np.float64)
    y_c = np.asarray(y_iq_4x, dtype=np.complex128)
    cs_re = CubicSpline(t_rx, y_c.real)
    cs_im = CubicSpline(t_rx, y_c.imag)
    idx = (PAD_SYMS * 4) + frac_offset + 4.0 * np.arange(PKT_LEN, dtype=np.float64)
    y_sym = cs_re(idx) + 1j * cs_im(idx)

    z = y_sym * np.conj(x_all_true)
    # Fine CFO around cfo_hint_hz +- 350 Hz (covers 1510 Hz beat instantaneous CFO on f708)
    n_grid = np.arange(PKT_LEN, dtype=np.float64)
    f_cands = np.linspace(cfo_hint_hz - 400.0, cfo_hint_hz + 400.0, 161)
    rot = np.exp(-2j * math.pi * f_cands[:, None] * n_grid[None, :] / FS_SYM)
    scores = np.abs(rot @ z)
    f_best = float(f_cands[int(np.argmax(scores))])

    # Complex quadratic envelope after derotating by f_best (tracks amplitude/phase beat smoothly)
    z_d = z * np.exp(-2j * math.pi * f_best * n_grid / FS_SYM)
    t_norm = (n_grid - (PKT_LEN - 1) / 2.0) / (PKT_LEN / 2.0)
    B = np.column_stack([np.ones(PKT_LEN), t_norm, t_norm ** 2])
    coef, _, _, _ = np.linalg.lstsq(B, z_d, rcond=None)
    env = B @ coef
    h_ref = env * np.exp(2j * math.pi * f_best * n_grid / FS_SYM)
    sig_pow = float(np.mean(np.abs(h_ref) ** 2))
    return h_ref, f_best, sig_pow


def extract_packets_for_capture(
    data_dir: Path,
    anchor: CaptureAnchor,
    sym_ref_even: np.ndarray,
    sym_ref_odd: np.ndarray | None = None,
    *,
    per_loop_counts: dict[str, tuple[int, int]] | None = None,
    use_raw_file1_rrc: bool = False,
    n_loops_use: int = 8,
) -> dict[str, dict[str, list[PacketSample]]]:
    """Slice strictly disjoint train/dev/test target & reference packets for one capture.

    Args:
        per_loop_counts: mapping split -> (n_target_per_loop, n_ref_per_loop).
            Defaults to train:(20, 8), dev:(12, 8), test:(20, 8) across 8 loops.
        use_raw_file1_rrc: if True, reads `file_1.dat` and applies our transparent 256-tap RRC
            matched filter locally around each packet (for end-to-end validation against file_2.dat).
    """
    sym_ref = sym_ref_odd if (anchor.bit_alignment == 1 and sym_ref_odd is not None) else sym_ref_even
    if per_loop_counts is None:
        per_loop_counts = {"train": (20, 8), "dev": (12, 8), "test": (20, 8)}

    cap_dir = data_dir / anchor.capture_id
    src_fname = "file_1.dat" if use_raw_file1_rrc else "file_2.dat"
    x_mm = np.memmap(cap_dir / src_fname, dtype=np.complex64, mode="r")
    n_tot = len(x_mm)
    h_rrc = make_rrc_taps() if use_raw_file1_rrc else None

    # Find the earliest loop where src_sym == 0 occurs after t = 1.45s (rx >= 2,900,000)
    # anchor_rx_pos corresponds to anchor_src_sym at ~t=2.0s.
    # Advance to next src_sym = 0 or back by anchor_src_sym:
    rx_loop0_start = anchor.anchor_rx_pos - anchor.anchor_src_sym * anchor.sps_eff
    while rx_loop0_start < 2_900_000.0:
        rx_loop0_start += SOURCE_N_SYMS * anchor.sps_eff

    out: dict[str, dict[str, list[PacketSample]]] = {
        sp: {"target": [], "ref": []} for sp in ("train", "dev", "test")
    }
    pkt_rx_len = (PKT_LEN + 2 * PAD_SYMS) * SPS_NOM
    rrc_margin = 512 if use_raw_file1_rrc else 0

    for loop_id in range(n_loops_use):
        rx_base = rx_loop0_start + loop_id * SOURCE_N_SYMS * anchor.sps_eff
        cluster_id = f"{anchor.capture_id}__L{loop_id:02d}"
        for split in ("train", "dev", "test"):
            n_tgt, n_ref = per_loop_counts[split]
            for role, count in (("target", n_tgt), ("ref", n_ref)):
                if count <= 0:
                    continue
                lo, hi = SPLIT_SRC_RANGES[split][role]
                usable = hi - lo - PKT_LEN
                stride = usable // count
                assert stride >= 2048, f"Stride too small ({stride}) for zero-overlap guarantee!"
                for k in range(count):
                    src_start = lo + k * stride + 128
                    rx_exact = rx_base + src_start * anchor.sps_eff
                    rx_int = int(math.floor(rx_exact))
                    frac = float(rx_exact - rx_int)
                    win_start = rx_int - PAD_SYMS * SPS_NOM
                    win_end = win_start + pkt_rx_len
                    if win_start - rrc_margin < 0 or win_end + rrc_margin >= n_tot:
                        continue

                    if use_raw_file1_rrc:
                        raw_seg = np.asarray(
                            x_mm[win_start - rrc_margin : win_end + rrc_margin], dtype=np.complex128
                        )
                        # GNU Radio's causal RRC filter in file_2.dat ALREADY carries the 127.5-sample
                        # filter group delay relative to file_1.dat; empirical cross-correlation verified
                        # lfilter(h_rrc, [1.0], file_1)[n] aligns with file_2[n+1] (|corr|=0.9775).
                        filt = signal.lfilter(h_rrc, [1.0], raw_seg)
                        y_seg = filt[rrc_margin - 1 : rrc_margin - 1 + pkt_rx_len].astype(np.complex64)
                    else:
                        y_seg = np.asarray(x_mm[win_start:win_end], dtype=np.complex64).copy()

                    x_true = sym_ref[src_start : src_start + PKT_LEN].copy()
                    x_known = x_true[KNOWN_IDX].copy()
                    h_ref, cfo_off, sig_pow = _offline_reference_channel(
                        y_seg, frac, x_true, anchor.offline_cfo_med_hz
                    )
                    uid = f"{anchor.capture_id}__L{loop_id:02d}__{split[0].upper()}{role[0].upper()}{k:03d}__S{src_start:06d}"
                    out[split][role].append(
                        PacketSample(
                            unit_id=uid,
                            capture_id=anchor.capture_id,
                            freq_mhz=anchor.freq_mhz,
                            nom_snr_db=anchor.nom_snr_db,
                            cluster_id=cluster_id,
                            loop_id=loop_id,
                            split=split,
                            role=role,
                            rx_start_sample=rx_int,
                            frac_rx_offset=frac,
                            src_sym_start=src_start,
                            time_sec=rx_exact / FS_RX,
                            bg_noise_var=anchor.bg_noise_var,
                            y_iq_4x=y_seg,
                            x_known=x_known,
                            x_all_true=x_true,
                            h_offline_ref=h_ref,
                            offline_cfo_hz=cfo_off,
                            offline_sig_pow=sig_pow,
                        )
                    )
    return out


def verify_zero_leakage(dataset: dict[str, dict[str, list[PacketSample]]]) -> dict[str, Any]:
    """Assert strictly disjoint source-symbol intervals and unit IDs across splits and target/ref."""
    ranges_seen: dict[tuple[str, str], set[int]] = {}
    all_uids: set[str] = set()
    counts: dict[str, int] = {}
    for sp in ("train", "dev", "test"):
        for role in ("target", "ref"):
            pkts = dataset[sp][role]
            counts[f"{sp}_{role}"] = len(pkts)
            sym_set: set[int] = set()
            for p in pkts:
                assert p.unit_id not in all_uids, f"Duplicate unit_id {p.unit_id}"
                all_uids.add(p.unit_id)
                lo, hi = SPLIT_SRC_RANGES[sp][role]
                assert lo <= p.src_sym_start and (p.src_sym_start + PKT_LEN) <= hi, (
                    f"Packet {p.unit_id} src_start={p.src_sym_start} outside [{lo}, {hi})"
                )
                sym_set.update(range(p.src_sym_start, p.src_sym_start + PKT_LEN))
            ranges_seen[(sp, role)] = sym_set

    keys = list(ranges_seen.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ov = len(ranges_seen[keys[i]] & ranges_seen[keys[j]])
            assert ov == 0, f"Source-symbol leakage detected between {keys[i]} and {keys[j]}: {ov} symbols!"
    return {"leakage_verified": True, "unique_unit_ids": len(all_uids), "counts": counts}


def apply_controlled_stress(
    pkt: PacketSample,
    rng: np.random.Generator,
    *,
    mode: str,
) -> PacketSample:
    """Create a controlled physical stress-test variant on top of real measured IQ waveform.

    Preserves real hardware channel & ISI while injecting controlled impairments:
      - `stress_low_snr`: inject calibrated complex AWGN so effective SNR drops by ~8-14 dB
        (brings snr20/snr10 into critical ~0-2 dB regime and snr0 into ~-3 dB regime)
      - `stress_cfo_phase_jump`: inject random CFO perturbation in [-600, +600] Hz + mid-packet
        phase jump in {-pi/2, +pi/2, pi} at symbol 128 (with probability 0.5)
      - `stress_burst_impairment`: inject combined extra noise + CFO offset + 12% burst impulse noise
    """
    y = np.asarray(pkt.y_iq_4x, dtype=np.complex128).copy()
    h_ref = pkt.h_offline_ref.copy()
    sig_pow = max(pkt.offline_sig_pow, 1e-18)
    n_rx = len(y)
    t_rx = (np.arange(n_rx, dtype=np.float64) - PAD_SYMS * SPS_NOM) / FS_RX
    t_sym = np.arange(PKT_LEN, dtype=np.float64) / FS_SYM
    new_bg = pkt.bg_noise_var
    meta = dict(pkt.extra_meta)

    if mode == "stress_low_snr":
        # Target post-RRC SNR ~ -1.5 dB for snr0, ~ +1.0 dB for snr10, ~ +2.0 dB for snr20
        target_snr_db = {20: 2.0, 10: 1.0, 0: -1.5}.get(pkt.nom_snr_db, 1.0)
        target_nv = sig_pow / (10.0 ** (target_snr_db / 10.0))
        add_nv = max(0.0, target_nv - pkt.bg_noise_var)
        # Apply RRC-shaped noise correlation or white noise? Inject white complex noise in 4x domain
        # scaled so symbol-rate variance increases by add_nv!
        # Wait: since y is already post-RRC, adding white noise of variance add_nv directly adds add_nv
        # to every decimated symbol sample!
        noise = math.sqrt(add_nv / 2.0) * (rng.standard_normal(n_rx) + 1j * rng.standard_normal(n_rx))
        y = y + noise
        new_bg = pkt.bg_noise_var + add_nv
        meta["stress_target_snr_db"] = target_snr_db
    elif mode == "stress_cfo_phase_jump":
        df_hz = float(rng.uniform(-650.0, 650.0))
        # Apply CFO rotation
        y = y * np.exp(2j * math.pi * df_hz * t_rx)
        h_ref = h_ref * np.exp(2j * math.pi * df_hz * t_sym)
        # With 50% probability, apply a phase jump after symbol 96 (between sync and mid-pilot!)
        jump_rad = 0.0
        if rng.random() < 0.5:
            jump_rad = float(rng.choice([-math.pi / 2.0, math.pi / 2.0, math.pi]))
            jump_rx = (PAD_SYMS + 96) * SPS_NOM
            y[jump_rx:] *= np.exp(1j * jump_rad)
            h_ref[96:] *= np.exp(1j * jump_rad)
        # Mild extra noise if snr20 so it's non-trivial
        if pkt.nom_snr_db == 20:
            add_nv = max(0.0, sig_pow / (10.0 ** (6.0 / 10.0)) - pkt.bg_noise_var)
            y = y + math.sqrt(add_nv / 2.0) * (rng.standard_normal(n_rx) + 1j * rng.standard_normal(n_rx))
            new_bg = pkt.bg_noise_var + add_nv
        meta["stress_df_hz"] = df_hz
        meta["stress_jump_rad"] = jump_rad
    elif mode == "stress_burst_impairment":
        df_hz = float(rng.uniform(-450.0, 450.0))
        y = y * np.exp(2j * math.pi * df_hz * t_rx)
        h_ref = h_ref * np.exp(2j * math.pi * df_hz * t_sym)
        target_snr_db = {20: 3.5, 10: 2.0, 0: -0.5}.get(pkt.nom_snr_db, 2.0)
        add_nv = max(0.0, sig_pow / (10.0 ** (target_snr_db / 10.0)) - pkt.bg_noise_var)
        y = y + math.sqrt(add_nv / 2.0) * (rng.standard_normal(n_rx) + 1j * rng.standard_normal(n_rx))
        new_bg = pkt.bg_noise_var + add_nv
        # Impulsive burst interference on 16 contiguous payload symbols
        b_sym = int(rng.integers(56, 176))
        b_rx0 = (PAD_SYMS + b_sym) * SPS_NOM
        b_rx1 = b_rx0 + 16 * SPS_NOM
        burst_nv = 4.0 * sig_pow
        y[b_rx0:b_rx1] += math.sqrt(burst_nv / 2.0) * (
            rng.standard_normal(b_rx1 - b_rx0) + 1j * rng.standard_normal(b_rx1 - b_rx0)
        )
        meta["stress_df_hz"] = df_hz
        meta["stress_burst_sym"] = b_sym
    else:
        raise ValueError(f"Unknown stress mode: {mode}")

    return PacketSample(
        unit_id=f"{pkt.unit_id}__{mode}",
        capture_id=pkt.capture_id,
        freq_mhz=pkt.freq_mhz,
        nom_snr_db=pkt.nom_snr_db,
        cluster_id=pkt.cluster_id,
        loop_id=pkt.loop_id,
        split=pkt.split,
        role=pkt.role,
        rx_start_sample=pkt.rx_start_sample,
        frac_rx_offset=pkt.frac_rx_offset,
        src_sym_start=pkt.src_sym_start,
        time_sec=pkt.time_sec,
        bg_noise_var=new_bg,
        y_iq_4x=y.astype(np.complex64),
        x_known=pkt.x_known.copy(),
        x_all_true=pkt.x_all_true.copy(),
        h_offline_ref=h_ref,
        offline_cfo_hz=pkt.offline_cfo_hz + float(meta.get("stress_df_hz", 0.0)),
        offline_sig_pow=sig_pow,
        perturbation_tag=mode,
        extra_meta=meta,
    )


def save_anchor_index(anchors: Sequence[CaptureAnchor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [a.__dict__ for a in anchors]
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
"""Controlled comparative experiments A (Global Content Addressing), B (Hard vs Posterior Fusion
vs Mixture Likelihood), and C (Lightweight Trainable Attention / Weight & Residual Network).

Strictly aligns with:
  - research/content_addressing_scalar/REPORT.md (Eqs. 1-9)
  - research/posterior_fusion/pages/01-08.md (orthogonal projection, total covariance, mixture LLR)
  - p12/P12_CONSISTENCY_FIX.md (single-count pilot evidence, zero target/holdout leakage)
"""
from __future__ import annotations

import math
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .adapter import (
    FS_SYM,
    HOLDOUT_IDX,
    KNOWN_IDX,
    PAD_SYMS,
    PAYLOAD_IDX,
    PILOT_IDX,
    PKT_LEN,
    SPS_NOM,
    SYNC_IDX,
    PacketSample,
)
from .baselines import (
    ClassicalConfig,
    ReceiverOutput,
    acquire_timing_and_cfo,
    extract_symbol_stream,
)
from .evaluator import (
    QPSK_CONSTELLATION,
    bits_to_qpsk,
    joint_mixture_qpsk_posterior,
    mixture_qpsk_posterior,
    qpsk_bit_llrs_gaussian,
    qpsk_hard_symbols,
    qpsk_to_bits,
)


# =============================================================================
# Shared Reference Pool & Pilot-Only Reference Extraction (Zero Leakage)
# =============================================================================

@dataclass
class RefEntry:
    """Pilot-only physical summary of one reference packet (never touches ref payload/holdout)."""
    unit_id: str
    capture_id: str  # evaluator-only audit metadata; NEVER read by receiver scoring!
    freq_mhz: int    # evaluator-only audit metadata
    nom_snr_db: int  # evaluator-only audit metadata
    time_sec: float
    est_timing_rx: float
    est_cfo_hz: float
    h_pow: float
    nv_est: float
    est_snr_lin: float
    est_snr_db: float
    slope_ratio: complex


def extract_reference_entry(pkt: PacketSample, cfg: ClassicalConfig) -> RefEntry:
    """Extract pilot-only physical statistics from a reference packet."""
    t_rx, cfo_hz, y_sym, _ = acquire_timing_and_cfo(pkt.y_iq_4x, pkt.x_known, cfg)
    n_idx = np.arange(PKT_LEN, dtype=np.float64)
    y_derot = y_sym * np.exp(-2j * math.pi * cfo_hz * n_idx / FS_SYM)
    yk = y_derot[KNOWN_IDX]
    xk = pkt.x_known
    n_pil = len(KNOWN_IDX)

    h_ls = complex(np.vdot(xk, yk) / max(float(np.vdot(xk, xk).real), 1e-18))
    resid = yk - h_ls * xk
    nv_resid = float(np.sum(np.abs(resid) ** 2) / max(n_pil - 2, 1))
    nv_bg = max(float(pkt.bg_noise_var), 1e-18)
    nv_est = max(cfg.noise_floor_mult * nv_bg, cfg.noise_blend * nv_resid + (1.0 - cfg.noise_blend) * nv_bg)
    h_pow = max(float(abs(h_ls) ** 2) - nv_est / float(n_pil), 0.05 * nv_est)
    snr_lin = h_pow / max(nv_est, 1e-18)

    # Normalized linear envelope slope from pilots
    t_k = (KNOWN_IDX.astype(np.float64) - (PKT_LEN - 1) / 2.0) / (PKT_LEN / 2.0)
    A = np.column_stack([xk, xk * t_k])
    reg = np.array([nv_est / max(h_pow, 1e-18), 4.0 * nv_est / max(h_pow, 1e-18)], dtype=np.float64)
    coef = np.linalg.solve(A.conj().T @ A + np.diag(reg), A.conj().T @ yk)
    slope_ratio = complex(coef[1] / (coef[0] + 1e-18))

    return RefEntry(
        unit_id=pkt.unit_id,
        capture_id=pkt.capture_id,
        freq_mhz=pkt.freq_mhz,
        nom_snr_db=pkt.nom_snr_db,
        time_sec=pkt.time_sec,
        est_timing_rx=t_rx,
        est_cfo_hz=cfo_hz,
        h_pow=h_pow,
        nv_est=nv_est,
        est_snr_lin=snr_lin,
        est_snr_db=10.0 * math.log10(max(snr_lin, 1e-12)),
        slope_ratio=slope_ratio,
    )


def build_cross_session_pool_for_target(
    target: PacketSample,
    refs_by_capture: dict[str, list[RefEntry]],
    *,
    per_capture: int = 4,
    rng: np.random.Generator,
) -> list[RefEntry]:
    """Build a balanced, shuffled global reference pool of M = 9 * per_capture references.

    Pool order is uniformly shuffled using `rng` independent of target packet content, so a
    fixed local window S = {0..k-1} satisfies the exact assumptions of content_addressing_scalar §2.
    """
    pool: list[RefEntry] = []
    for cap_id in sorted(refs_by_capture.keys()):
        entries = refs_by_capture[cap_id]
        if len(entries) <= per_capture:
            chosen = list(entries)
        else:
            idx = rng.choice(len(entries), size=per_capture, replace=False)
            chosen = [entries[int(i)] for i in idx]
        pool.extend(chosen)
    perm = rng.permutation(len(pool))
    return [pool[int(i)] for i in perm]


def build_intra_session_pool_for_target(
    target: PacketSample,
    refs_same_capture: list[RefEntry],
    *,
    pool_size: int = 24,
) -> list[RefEntry]:
    """Build an intra-session temporal pool sorted by time distance |t_ref - t_target|.

    Thus local window S = {0..k-1} selects the k temporally nearest reference packets,
    while global methods access all M=pool_size packets spanning distant loops/times.
    """
    ordered = sorted(refs_same_capture, key=lambda r: abs(r.time_sec - target.time_sec))
    return ordered[:pool_size]


# =============================================================================
# Experiment A: Global Reference Pool & Content Addressing
# =============================================================================

def pilot_marginal_loglik_for_ref(
    y_iq_4x: np.ndarray,
    x_known: np.ndarray,
    ref: RefEntry,
    *,
    y_sym_cache: dict[float, np.ndarray] | None = None,
) -> tuple[float, float]:
    """Exact phase-marginalized complex Gaussian log-likelihood of target pilots under ref prior.

    Model under ref j:
        y_k = h * x_k * exp(j 2pi f_j k / fsym) + n_k,  h ~ CN(0, P_j),  n_k ~ CN(0, nv_j).
    Marginalizing out complex h yields exact closed-form log p(y_known | x_known, ref_j):
        ll = -N_p*log(pi*nv_j) - log(1 + N_p*P_j/nv_j) - ||y||^2/nv_j + |<z, 1>|^2 / (nv_j*(N_p + nv_j/P_j))
    Returns (log_lik, coherent_corr_power).
    """
    t_key = round(ref.est_timing_rx, 2)
    if y_sym_cache is not None and t_key in y_sym_cache:
        y_sym = y_sym_cache[t_key]
    else:
        y_sym = extract_symbol_stream(y_iq_4x, ref.est_timing_rx)
        if y_sym_cache is not None:
            y_sym_cache[t_key] = y_sym

    yk = y_sym[KNOWN_IDX]
    n_p = float(len(KNOWN_IDX))
    rot = np.exp(-2j * math.pi * ref.est_cfo_hz * KNOWN_IDX.astype(np.float64) / FS_SYM)
    z = yk * np.conj(x_known) * rot
    coh = abs(complex(np.sum(z))) ** 2
    y_e = float(np.sum(np.abs(yk) ** 2))
    p_j = max(ref.h_pow, 1e-18)
    nv_j = max(ref.nv_est, 1e-18)
    ll = (
        -n_p * math.log(math.pi * nv_j)
        - math.log(1.0 + n_p * p_j / nv_j)
        - y_e / nv_j
        + coh / (nv_j * (n_p + nv_j / p_j))
    )
    return float(ll), float(coh / (n_p * n_p * p_j + 1e-30))


def _softmax(scores: np.ndarray, temp: float = 1.0) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64) / max(float(temp), 1e-6)
    s = s - float(np.max(s))
    e = np.exp(np.clip(s, -500.0, 0.0))
    return e / float(np.sum(e))


def run_prior_conditioned_receiver(
    pkt: PacketSample,
    cfg: ClassicalConfig,
    *,
    cfo_prior_mean: float,
    cfo_prior_var: float,
    h_pow_prior: float,
    nv_prior: float,
    timing_prior: float,
    prior_strength: float = 1.0,
    extra: dict[str, Any] | None = None,
) -> ReceiverOutput:
    """Common downstream MAP-CFO + MMSE receiver conditioned on reference-aggregated prior.

    All arms in Experiment A (local window, global uniform, global SNR weight, global content
    matching, learned attention) use this EXACT same downstream estimator so differences isolate
    purely the reference scope and weighting mechanism.
    """
    t0 = time.perf_counter()
    # Timing: blend target pilot timing search lightly around prior or full grid
    t_rx_local, cfo_local, y_sym_local, _ = acquire_timing_and_cfo(pkt.y_iq_4x, pkt.x_known, cfg)
    # Decide timing: if prior variance on CFO is tight (coherent match), test timing_prior vs t_rx_local
    y_sym_pr = extract_symbol_stream(pkt.y_iq_4x, timing_prior)
    # Evaluate MAP objective over fine CFO grid combining pilot likelihood + Gaussian CFO prior:
    #   J(f) = |sum_k y_k x_k^* e^{-j2pi f k/fsym}|^2 / (nv_eff * (N_p + nv_eff/P_h)) - (f - f_pr)^2 / (2 * v_f)
    nv_bg = max(float(pkt.bg_noise_var), 1e-18)
    nv_eff_prior = max(0.5 * nv_bg, 0.65 * nv_prior + 0.35 * nv_bg)
    p_h_eff = max(h_pow_prior, 0.05 * nv_eff_prior)

    # Choose between local timing and prior timing by MAP objective
    known_f = KNOWN_IDX.astype(np.float64)
    n_p = float(len(KNOWN_IDX))
    v_f = max(cfo_prior_var, 15.0 ** 2) / max(prior_strength, 1e-6)

    # Search grid covering both prior neighborhood and local peak
    f_cands = np.unique(
        np.concatenate(
            [
                np.linspace(cfo_prior_mean - 450.0, cfo_prior_mean + 450.0, 91),
                np.linspace(cfo_local - 350.0, cfo_local + 350.0, 71),
            ]
        )
    )
    rot_mat = np.exp(-2j * math.pi * f_cands[:, None] * known_f[None, :] / FS_SYM)
    prior_pen = -0.5 * ((f_cands - cfo_prior_mean) ** 2) / v_f

    best_obj = -1e300
    best_cfo = cfo_prior_mean
    best_t_rx = t_rx_local
    best_y_sym = y_sym_local
    for cand_t, cand_y in ((t_rx_local, y_sym_local), (timing_prior, y_sym_pr)):
        zk = cand_y[KNOWN_IDX] * np.conj(pkt.x_known)
        coh2 = np.abs(rot_mat @ zk) ** 2
        ll_f = coh2 / (nv_eff_prior * (n_p + nv_eff_prior / p_h_eff))
        obj = ll_f + prior_pen
        idx_b = int(np.argmax(obj))
        if float(obj[idx_b]) > best_obj:
            best_obj = float(obj[idx_b])
            best_cfo = float(f_cands[idx_b])
            best_t_rx = cand_t
            best_y_sym = cand_y

    # MMSE channel estimate at best_cfo using reference-informed signal power & noise variance
    n_idx = np.arange(PKT_LEN, dtype=np.float64)
    rot_all = np.exp(2j * math.pi * best_cfo * n_idx / FS_SYM)
    y_derot = best_y_sym * np.conj(rot_all)
    yk = y_derot[KNOWN_IDX]
    xk = pkt.x_known
    h_ls = complex(np.vdot(xk, yk) / n_p)
    resid = yk - h_ls * xk
    nv_resid = float(np.sum(np.abs(resid) ** 2) / max(int(n_p) - 2, 1))
    nv_est = max(
        cfg.noise_floor_mult * nv_bg,
        0.25 * nv_resid + 0.45 * nv_prior + 0.30 * nv_bg,
    )
    lam = p_h_eff / (p_h_eff + nv_est / n_p)
    h_mmse = lam * h_ls
    h_var_post = (p_h_eff * (nv_est / n_p)) / (p_h_eff + nv_est / n_p)
    # Include residual CFO variance contribution to effective predictive variance across packet
    dt_center = (n_idx - (PKT_LEN - 1) / 2.0) / FS_SYM
    # Posterior CFO variance combines Fisher information from pilots + prior precision
    snr_pil = p_h_eff / max(nv_est, 1e-18)
    t_rms2 = float(np.mean(((known_f - np.mean(known_f)) / FS_SYM) ** 2))
    fisher_f = 2.0 * (2.0 * math.pi) ** 2 * n_p * snr_pil * t_rms2
    post_var_f = 1.0 / (fisher_f + 1.0 / v_f)
    phase_var_n = ((2.0 * math.pi * dt_center) ** 2) * post_var_f
    nv_eff = (nv_est + h_var_post + (abs(h_mmse) ** 2) * np.minimum(phase_var_n, 0.5)) * cfg.var_calib_scale

    h_hat = np.full(PKT_LEN, h_mmse, dtype=np.complex128) * rot_all
    x_eq = np.conj(h_hat) * best_y_sym / (np.abs(h_hat) ** 2 + nv_eff)
    bit_llrs = qpsk_bit_llrs_gaussian(best_y_sym, h_hat, nv_eff)
    hard_syms = qpsk_hard_symbols(x_eq)

    out_extra = {
        "cfo_prior_mean": cfo_prior_mean,
        "cfo_prior_std": math.sqrt(max(cfo_prior_var, 0.0)),
        "h_pow_prior": h_pow_prior,
    }
    if extra:
        out_extra.update(extra)
    return ReceiverOutput(
        y_sym=best_y_sym,
        h_hat=h_hat,
        nv_hat=nv_eff,
        x_eq=x_eq,
        bit_llrs=bit_llrs,
        hard_syms=hard_syms,
        est_timing_rx=best_t_rx,
        est_cfo_hz=best_cfo,
        est_noise_var=nv_est,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        extra=out_extra,
    )


def score_reference_pool(
    pkt: PacketSample,
    pool: Sequence[RefEntry],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact pilot marginal log-likelihood scores and normalized correlations for all M refs."""
    cache: dict[float, np.ndarray] = {}
    scores = np.empty(len(pool), dtype=np.float64)
    corrs = np.empty(len(pool), dtype=np.float64)
    for j, ref in enumerate(pool):
        ll, cr = pilot_marginal_loglik_for_ref(pkt.y_iq_4x, pkt.x_known, ref, y_sym_cache=cache)
        scores[j] = ll
        corrs[j] = cr
    return scores, corrs


def aggregate_ref_weights(
    pool: Sequence[RefEntry],
    weights: np.ndarray,
    *,
    cfo_var_floor: float = 40.0 ** 2,
) -> dict[str, float]:
    """Aggregate reference pool values using probability weights alpha (summing to 1)."""
    w = np.asarray(weights, dtype=np.float64)
    w = np.maximum(w, 0.0)
    w /= max(float(np.sum(w)), 1e-18)
    cfos = np.array([r.est_cfo_hz for r in pool], dtype=np.float64)
    pows = np.array([r.h_pow for r in pool], dtype=np.float64)
    nvs = np.array([r.nv_est for r in pool], dtype=np.float64)
    tims = np.array([r.est_timing_rx for r in pool], dtype=np.float64)

    f_mean = float(np.sum(w * cfos))
    f_var = float(np.sum(w * ((cfos - f_mean) ** 2))) + cfo_var_floor
    # Geometric mean for positive power/noise spans across 20 dB dynamic range
    p_mean = float(np.exp(np.sum(w * np.log(np.maximum(pows, 1e-18)))))
    nv_mean = float(np.exp(np.sum(w * np.log(np.maximum(nvs, 1e-18)))))
    t_mean = float(np.sum(w * tims))
    return {
        "cfo_prior_mean": f_mean,
        "cfo_prior_var": f_var,
        "h_pow_prior": p_mean,
        "nv_prior": nv_mean,
        "timing_prior": t_mean,
    }


def run_experiment_a_arms(
    pkt: PacketSample,
    pool: Sequence[RefEntry],
    cfg: ClassicalConfig,
    *,
    local_k: int = 4,
    content_temp: float = 1.5,
    snr_temp: float = 5.0,
) -> tuple[dict[str, ReceiverOutput], dict[str, Any]]:
    """Run all paired arms of Experiment A on one packet with identical reference pool."""
    M = len(pool)
    scores, _ = score_reference_pool(pkt, pool)
    snr_dbs = np.array([r.est_snr_db for r in pool], dtype=np.float64)

    # Weight vectors for each arm (all sum to 1)
    w_local_k = np.zeros(M, dtype=np.float64)
    w_local_k[: min(local_k, M)] = 1.0 / float(min(local_k, M))

    w_unif = np.full(M, 1.0 / float(M), dtype=np.float64)
    w_snr = _softmax(snr_dbs, temp=snr_temp)
    w_content = _softmax(scores, temp=content_temp)
    w_cold = np.zeros(M, dtype=np.float64)
    w_cold[int(np.argmax(scores))] = 1.0
    w_hot = _softmax(scores, temp=content_temp * 15.0)

    # Audit metrics (offline only: same-capture weight mass & content addressing theory terms)
    same_cap_mask = np.array([1.0 if r.capture_id == pkt.capture_id else 0.0 for r in pool], dtype=np.float64)
    same_freq_mask = np.array([1.0 if r.freq_mhz == pkt.freq_mhz else 0.0 for r in pool], dtype=np.float64)
    ref_cfos = np.array([r.est_cfo_hz for r in pool], dtype=np.float64)
    # Effective gap between best same-capture score and best mismatched-capture score
    if np.any(same_cap_mask > 0.5) and np.any(same_cap_mask < 0.5):
        s_match = float(np.max(scores[same_cap_mask > 0.5]))
        s_mismatch = float(np.max(scores[same_cap_mask < 0.5]))
        score_margin = s_match - s_mismatch
    else:
        score_margin = float("nan")

    arms_spec = [
        ("A1_local_window_k4", w_local_k, 1.0),
        ("A2_global_uniform", w_unif, 1.0),
        ("A3_global_snr_weight", w_snr, 1.0),
        ("A4_global_content_attn", w_content, 1.0),
        ("A5_content_cold_hard", w_cold, 1.0),
        ("A6_content_hot", w_hot, 1.0),
    ]

    outputs: dict[str, ReceiverOutput] = {}
    audit: dict[str, Any] = {
        "unit_id": pkt.unit_id,
        "capture_id": pkt.capture_id,
        "nom_snr_db": pkt.nom_snr_db,
        "freq_mhz": pkt.freq_mhz,
        "offline_cfo_hz": pkt.offline_cfo_hz,
        "score_margin_match_vs_mismatch": score_margin,
        "pool_M": M,
        "local_k": local_k,
        "same_cap_in_local_k": int(np.sum(same_cap_mask[:local_k])),
        "same_cap_total": int(np.sum(same_cap_mask)),
    }

    for name, w, p_str in arms_spec:
        agg = aggregate_ref_weights(pool, w)
        hit_cap = float(np.sum(w * same_cap_mask))
        hit_freq = float(np.sum(w * same_freq_mask))
        cfo_prior_err2 = (agg["cfo_prior_mean"] - pkt.offline_cfo_hz) ** 2
        # L2 weight distance to ideal uniform mass over same-capture refs (for Eq. 2 check)
        n_sc = float(np.sum(same_cap_mask))
        e_ideal = same_cap_mask / max(n_sc, 1.0)
        w_l2_err2 = float(np.sum((w - e_ideal) ** 2))
        w_var_sum = float(np.sum((w - 1.0 / M) ** 2))
        out = run_prior_conditioned_receiver(
            pkt,
            cfg,
            cfo_prior_mean=agg["cfo_prior_mean"],
            cfo_prior_var=agg["cfo_prior_var"],
            h_pow_prior=agg["h_pow_prior"],
            nv_prior=agg["nv_prior"],
            timing_prior=agg["timing_prior"],
            prior_strength=p_str,
            extra={
                "same_cap_weight": hit_cap,
                "same_freq_weight": hit_freq,
                "cfo_prior_abs_err": abs(agg["cfo_prior_mean"] - pkt.offline_cfo_hz),
            },
        )
        outputs[name] = out
        audit[f"{name}__hit_cap"] = hit_cap
        audit[f"{name}__hit_freq"] = hit_freq
        audit[f"{name}__cfo_prior_err2"] = cfo_prior_err2
        audit[f"{name}__w_l2_err2"] = w_l2_err2
        audit[f"{name}__w_var_sum"] = w_var_sum
        audit[f"{name}__ref_cfo_est"] = float(np.sum(w * ref_cfos))
    return outputs, audit


# =============================================================================
# Experiment B: Common Physical Candidate Bank — Hard vs Linear Fusion vs Mixture
# =============================================================================

@dataclass
class CandidateBank:
    """Shared L-hypothesis physical candidate bank for one packet (strictly pilot-only)."""
    y_sym: np.ndarray        # (256,) complex128
    timing_rx: float
    h_cands: np.ndarray      # (L, 256) complex128 per-candidate channel predictions
    nv_cands: np.ndarray     # (L, 256) float64 per-candidate effective predictive variance
    log_ev: np.ndarray       # (L,) float64 exact single-count pilot log-evidence + log-prior
    cand_cfos: np.ndarray    # (L,) float64 per-candidate CFO
    cand_feats: np.ndarray   # (L, 12) float64 strictly pilot-observable features for Exp C
    cand_names: tuple[str, ...]
    build_ms: float


def _fit_scalar_cfo_branch(
    y_sym: np.ndarray,
    x_known: np.ndarray,
    cfo_hz: float,
    p_h_prior: float,
    nv_est: float,
    var_calib: float,
    *,
    extra_cfo_var: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, float]]:
    """Fit complex scalar MMSE channel at fixed CFO hypothesis and return exact marginal log-evidence.

    Exact complex Gaussian marginal log-likelihood of y_known given x_known & h ~ CN(0, p_h_prior):
        log p(y_k | x_k, f) = -N_p log(pi nv) - log(1 + N_p p_h / nv) - ||y_k||^2/nv + |z_sum|^2 / (nv*(N_p + nv/p_h))
    Single-count: pilots are scored ONCE via exact marginal evidence (no double counting).
    """
    n_idx = np.arange(PKT_LEN, dtype=np.float64)
    rot_all = np.exp(2j * math.pi * cfo_hz * n_idx / FS_SYM)
    y_derot = y_sym * np.conj(rot_all)
    yk = y_derot[KNOWN_IDX]
    n_p = float(len(KNOWN_IDX))
    z_sum = complex(np.vdot(x_known, yk))
    y_e = float(np.sum(np.abs(yk) ** 2))
    p_h = max(p_h_prior, 1e-18)
    nv = max(nv_est, 1e-18)

    log_ev = (
        -n_p * math.log(math.pi * nv)
        - math.log(1.0 + n_p * p_h / nv)
        - y_e / nv
        + (abs(z_sum) ** 2) / (nv * (n_p + nv / p_h))
    )
    post_var_h = (p_h * (nv / n_p)) / (p_h + nv / n_p)
    h_post = (p_h / (p_h * n_p + nv)) * z_sum
    h_vec = np.full(PKT_LEN, h_post, dtype=np.complex128) * rot_all

    # Separate sync vs mid+tail residuals (strictly pilot-observable diagnostic features)
    n_sync = len(SYNC_IDX)
    res_all = yk - h_post * x_known
    res_sync = float(np.mean(np.abs(res_all[:n_sync]) ** 2) / nv)
    res_tail = float(np.mean(np.abs(res_all[n_sync:]) ** 2) / nv)
    # Phase consistency between first half (sync) and second half (mid+tail)
    h_pre = complex(np.vdot(x_known[:n_sync], yk[:n_sync]) / float(n_sync))
    h_post_half = complex(np.vdot(x_known[n_sync:], yk[n_sync:]) / float(len(KNOWN_IDX) - n_sync))
    ph_diff = abs(math.atan2((h_post_half * np.conj(h_pre)).imag, (h_post_half * np.conj(h_pre)).real))

    dt = (n_idx - (PKT_LEN - 1) / 2.0) / FS_SYM
    nv_vec = (nv + post_var_h + (abs(h_post) ** 2) * ((2.0 * math.pi * dt) ** 2) * extra_cfo_var) * var_calib
    feats = {
        "res_sync": res_sync,
        "res_tail": res_tail,
        "ph_diff": ph_diff,
        "h_abs": abs(h_post),
        "post_var_ratio": post_var_h / nv,
    }
    return h_vec, nv_vec, float(log_ev), feats


def _fit_slope_branch(
    y_sym: np.ndarray,
    x_known: np.ndarray,
    cfo_hz: float,
    p_h_prior: float,
    nv_est: float,
    slope_reg: float,
    var_calib: float,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, float]]:
    """Fit 2-DoF complex affine channel h_env(t) = c0 + c1*t with exact marginal log-evidence."""
    n_idx = np.arange(PKT_LEN, dtype=np.float64)
    rot_all = np.exp(2j * math.pi * cfo_hz * n_idx / FS_SYM)
    y_derot = y_sym * np.conj(rot_all)
    yk = y_derot[KNOWN_IDX]
    t_all = (n_idx - (PKT_LEN - 1) / 2.0) / (PKT_LEN / 2.0)
    t_k = t_all[KNOWN_IDX]
    A = np.column_stack([x_known, x_known * t_k])  # (16, 2)
    p_h = max(p_h_prior, 1e-18)
    nv = max(nv_est, 1e-18)
    v_prior = np.array([p_h, p_h / max(slope_reg, 0.2)], dtype=np.float64)
    P_inv = np.diag(1.0 / v_prior)
    # Posterior precision M = P_inv + A^H A / nv
    M_prec = P_inv + (A.conj().T @ A) / nv
    cov_c = np.linalg.inv(M_prec)
    rhs = (A.conj().T @ yk) / nv
    coef = cov_c @ rhs
    # Exact complex Gaussian marginal log-likelihood via matrix determinant & Woodbury:
    # log p(y) = -N_p log(pi nv) - log det(I + diag(v_prior) A^H A / nv) - ||y||^2/nv + rhs^H cov_c rhs
    sign, logdet_m = np.linalg.slogdet(M_prec)
    logdet_p_inv = float(np.sum(np.log(1.0 / v_prior)))
    log_ev = (
        -len(KNOWN_IDX) * math.log(math.pi * nv)
        - float(logdet_m - logdet_p_inv)
        - float(np.sum(np.abs(yk) ** 2)) / nv
        + float(np.real(np.vdot(rhs, coef)))
    )
    B_all = np.column_stack([np.ones(PKT_LEN), t_all])
    h_env = B_all @ coef
    ch_var = np.maximum(np.real(np.einsum("ni,ij,nj->n", B_all, cov_c, B_all)), 0.0)
    h_vec = h_env * rot_all
    nv_vec = (nv + ch_var) * var_calib

    n_sync = len(SYNC_IDX)
    res_all = yk - (A @ coef)
    feats = {
        "res_sync": float(np.mean(np.abs(res_all[:n_sync]) ** 2) / nv),
        "res_tail": float(np.mean(np.abs(res_all[n_sync:]) ** 2) / nv),
        "ph_diff": float(abs(coef[1]) / (abs(coef[0]) + 1e-18)),
        "h_abs": float(abs(coef[0])),
        "post_var_ratio": float(np.mean(ch_var) / nv),
    }
    return h_vec, nv_vec, float(log_ev), feats


def _fit_piecewise_jump_branch(
    y_sym: np.ndarray,
    x_known: np.ndarray,
    cfo_hz: float,
    p_h_prior: float,
    nv_est: float,
    split_sym: int,
    var_calib: float,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, float]]:
    """Fit piecewise channel allowing a mid-packet phase/gain jump at `split_sym` (exact marginal evidence)."""
    n_idx = np.arange(PKT_LEN, dtype=np.float64)
    rot_all = np.exp(2j * math.pi * cfo_hz * n_idx / FS_SYM)
    y_derot = y_sym * np.conj(rot_all)
    yk = y_derot[KNOWN_IDX]
    m_pre = KNOWN_IDX < split_sym
    m_post = ~m_pre
    p_h = max(p_h_prior, 1e-18)
    nv = max(nv_est, 1e-18)

    def _seg_fit(ys: np.ndarray, xs: np.ndarray) -> tuple[complex, float, float]:
        ns = float(len(xs))
        zs = complex(np.vdot(xs, ys))
        ye = float(np.sum(np.abs(ys) ** 2))
        ev = -ns * math.log(math.pi * nv) - math.log(1.0 + ns * p_h / nv) - ye / nv + (abs(zs) ** 2) / (nv * (ns + nv / p_h))
        h_s = (p_h / (p_h * ns + nv)) * zs
        v_s = (p_h * (nv / ns)) / (p_h + nv / ns)
        return h_s, v_s, float(ev)

    h1, v1, ev1 = _seg_fit(yk[m_pre], x_known[m_pre])
    h2, v2, ev2 = _seg_fit(yk[m_post], x_known[m_post])
    # Exact joint marginal evidence of independent pre/post channel gains (naturally pays Occam penalty for 2nd gain!)
    log_ev = ev1 + ev2
    h_env = np.where(n_idx < split_sym, h1, h2)
    ch_var = np.where(n_idx < split_sym, v1, v2)
    h_vec = h_env * rot_all
    nv_vec = (nv + ch_var) * var_calib

    res_pre = float(np.mean(np.abs(yk[m_pre] - h1 * x_known[m_pre]) ** 2) / nv)
    res_post = float(np.mean(np.abs(yk[m_post] - h2 * x_known[m_post]) ** 2) / nv)
    ph_jump = abs(math.atan2((h2 * np.conj(h1)).imag, (h2 * np.conj(h1)).real))
    feats = {
        "res_sync": res_pre,
        "res_tail": res_post,
        "ph_diff": ph_jump,
        "h_abs": 0.5 * (abs(h1) + abs(h2)),
        "post_var_ratio": 0.5 * (v1 + v2) / nv,
    }
    return h_vec, nv_vec, float(log_ev), feats


def build_common_candidate_bank(
    pkt: PacketSample,
    pool: Sequence[RefEntry],
    cfg: ClassicalConfig,
    *,
    content_temp: float = 1.5,
) -> CandidateBank:
    """Build the unified L=12 physical candidate bank shared identically by Exp B & Exp C.

    Strictly pilot-only: reads ONLY pkt.y_iq_4x, pkt.x_known, pkt.bg_noise_var, and pilot-only RefEntries.
    Candidates (L = 12):
      0..6 : 7 multi-hypothesis CFO grid around local pilot ML peak: {-900,-600,-300,0,+300,+600,+900} Hz
      7    : Global content-matched Top-1 cluster CFO + power/noise prior branch
      8    : Global content-matched Top-2 distinct-CFO cluster prior branch
      9    : Linear intra-packet drift (2-DoF affine slope) branch at best content/local CFO
      10   : Piecewise phase-jump branch (split at sym 96, between sync and mid-pilot) at local CFO
      11   : Piecewise phase-jump branch (split at sym 96) at content-matched prior CFO
    """
    t0 = time.perf_counter()
    timing_rx, cfo_local, y_sym, _ = acquire_timing_and_cfo(pkt.y_iq_4x, pkt.x_known, cfg)
    n_idx = np.arange(PKT_LEN, dtype=np.float64)

    # Local pilot power & noise baseline
    yk_raw = y_sym[KNOWN_IDX]
    nv_bg = max(float(pkt.bg_noise_var), 1e-18)
    y_derot_loc = y_sym * np.exp(-2j * math.pi * cfo_local * n_idx / FS_SYM)
    h_loc = complex(np.vdot(pkt.x_known, y_derot_loc[KNOWN_IDX]) / len(KNOWN_IDX))
    nv_res_loc = float(np.sum(np.abs(y_derot_loc[KNOWN_IDX] - h_loc * pkt.x_known) ** 2) / (len(KNOWN_IDX) - 2))
    nv_loc = max(cfg.noise_floor_mult * nv_bg, cfg.noise_blend * nv_res_loc + (1.0 - cfg.noise_blend) * nv_bg)
    p_h_loc = max(float(np.mean(np.abs(yk_raw) ** 2)) - nv_loc, 0.05 * nv_loc)

    # Reference pool content scores for global-cluster candidates (7, 8, 9, 11)
    if len(pool) > 0:
        ref_scores, _ = score_reference_pool(pkt, pool)
        w_ref = _softmax(ref_scores, temp=content_temp)
        agg1 = aggregate_ref_weights(pool, w_ref)
        cfo_ref1 = agg1["cfo_prior_mean"]
        p_h_ref1 = agg1["h_pow_prior"]
        nv_ref1 = max(0.5 * nv_bg, 0.5 * agg1["nv_prior"] + 0.5 * nv_bg)
        # Top-2 distinct CFO cluster (mask references within 180 Hz of cfo_ref1)
        ref_cfos = np.array([r.est_cfo_hz for r in pool], dtype=np.float64)
        far_mask = np.abs(ref_cfos - cfo_ref1) > 180.0
        if np.any(far_mask):
            s2 = np.where(far_mask, ref_scores, -1e9)
            w_ref2 = _softmax(s2, temp=content_temp)
            agg2 = aggregate_ref_weights(pool, w_ref2)
            cfo_ref2 = agg2["cfo_prior_mean"]
            p_h_ref2 = agg2["h_pow_prior"]
            nv_ref2 = max(0.5 * nv_bg, 0.5 * agg2["nv_prior"] + 0.5 * nv_bg)
        else:
            cfo_ref2, p_h_ref2, nv_ref2 = cfo_local + 250.0, p_h_loc, nv_loc
    else:
        cfo_ref1, p_h_ref1, nv_ref1 = cfo_local, p_h_loc, nv_loc
        cfo_ref2, p_h_ref2, nv_ref2 = cfo_local + 250.0, p_h_loc, nv_loc

    # Shared noise scale `nv_common` for apples-to-apples evidence comparison across all 12 candidates
    nv_common = max(cfg.noise_floor_mult * nv_bg, 0.30 * nv_res_loc + 0.35 * nv_ref1 + 0.35 * nv_bg)
    p_h_common = max(0.5 * p_h_loc + 0.5 * p_h_ref1, 0.05 * nv_common)

    # Extract top-3 genuinely competing local CFO spectral peaks from pilot periodogram (min separation 240 Hz)
    f_scan = np.linspace(cfg.cfo_search_min_hz, cfg.cfo_search_max_hz, 161)
    known_f = KNOWN_IDX.astype(np.float64)
    zk_raw = y_sym[KNOWN_IDX] * np.conj(pkt.x_known)
    spec_p = np.abs(np.exp(-2j * math.pi * f_scan[:, None] * known_f[None, :] / FS_SYM) @ zk_raw)
    peak_cfos: list[float] = [cfo_local]
    spec_work = spec_p.copy()
    spec_work[np.abs(f_scan - cfo_local) < 240.0] = -1.0
    for _ in range(2):
        pi = int(np.argmax(spec_work))
        if spec_work[pi] > 0.0:
            fc = float(f_scan[pi])
            peak_cfos.append(fc)
            spec_work[np.abs(f_scan - fc) < 240.0] = -1.0
        else:
            peak_cfos.append(cfo_local + 350.0 * len(peak_cfos))

    # Build L=12 genuinely competing physical hypotheses:
    #   0..2 : Top-3 local pilot spectral CFO peaks (captures sidelobe/grating-lobe ambiguity in low SNR!)
    #   3..4 : Fine +-180 Hz neighborhood around primary local peak (tracks phase drift uncertainty)
    #   5..6 : Global content Top-1 & Top-2 distinct-CFO cluster priors (from cross-session pool)
    #   7    : Global content Top-3 distinct-CFO cluster prior (or local + 450 Hz fallback)
    #   8    : 2-DoF intra-packet linear drift (affine slope) branch at content-matched CFO (tracks f708 1510Hz beat)
    #   9    : 2-DoF affine slope branch at local ML CFO
    #  10..11: Piecewise mid-packet phase-jump branches (split at sym=96) at local CFO & content prior CFO
    if len(pool) > 0:
        ref_cfos_all = np.array([r.est_cfo_hz for r in pool], dtype=np.float64)
        far3 = (np.abs(ref_cfos_all - cfo_ref1) > 180.0) & (np.abs(ref_cfos_all - cfo_ref2) > 180.0)
        if np.any(far3):
            s3 = np.where(far3, ref_scores, -1e9)
            w_ref3 = _softmax(s3, temp=content_temp)
            agg3 = aggregate_ref_weights(pool, w_ref3)
            cfo_ref3, p_h_ref3 = agg3["cfo_prior_mean"], agg3["h_pow_prior"]
        else:
            cfo_ref3, p_h_ref3 = cfo_ref1 - 220.0, p_h_ref1
    else:
        cfo_ref3, p_h_ref3 = cfo_local - 220.0, p_h_loc

    cand_h_list: list[np.ndarray] = []
    cand_nv_list: list[np.ndarray] = []
    cand_ev_list: list[float] = []
    cand_cfo_list: list[float] = []
    cand_names: list[str] = []
    raw_feats: list[dict[str, float]] = []
    family_ids: list[int] = []

    # Family 0: 5 multi-peak local CFO branches (total family prior 0.35 -> 0.07 each)
    local_cfo_specs = [
        ("cfo_peak1", peak_cfos[0]),
        ("cfo_peak2", peak_cfos[1]),
        ("cfo_peak3", peak_cfos[2]),
        ("cfo_loc_m180", cfo_local - 180.0),
        ("cfo_loc_p180", cfo_local + 180.0),
    ]
    for nm, f_c in local_cfo_specs:
        h_v, nv_v, ev, ft = _fit_scalar_cfo_branch(
            y_sym, pkt.x_known, f_c, p_h_common, nv_common, cfg.var_calib_scale, extra_cfo_var=50.0 ** 2
        )
        cand_h_list.append(h_v)
        cand_nv_list.append(nv_v)
        cand_ev_list.append(ev + math.log(0.35 / len(local_cfo_specs)))
        cand_cfo_list.append(f_c)
        cand_names.append(nm)
        raw_feats.append(ft)
        family_ids.append(0)

    # Family 1: Global content Top-1, Top-2, Top-3 reference cluster branches (total prior 0.30)
    for nm, f_c, p_c, pr_w in (
        ("global_ref_top1", cfo_ref1, p_h_ref1, 0.16),
        ("global_ref_top2", cfo_ref2, p_h_ref2, 0.08),
        ("global_ref_top3", cfo_ref3, p_h_ref3, 0.06),
    ):
        h_v, nv_v, ev, ft = _fit_scalar_cfo_branch(
            y_sym, pkt.x_known, f_c, max(p_c, 0.05 * nv_common), nv_common, cfg.var_calib_scale, extra_cfo_var=35.0 ** 2
        )
        cand_h_list.append(h_v)
        cand_nv_list.append(nv_v)
        cand_ev_list.append(ev + math.log(pr_w))
        cand_cfo_list.append(f_c)
        cand_names.append(nm)
        raw_feats.append(ft)
        family_ids.append(1)

    # Family 2: Intra-packet linear slope branches (total prior 0.15 -> 0.075 each)
    for nm, f_c in (("slope_drift_ref", cfo_ref1), ("slope_drift_loc", cfo_local)):
        h_sl, nv_sl, ev_sl, ft_sl = _fit_slope_branch(
            y_sym, pkt.x_known, f_c, p_h_common, nv_common, cfg.slope_reg, cfg.var_calib_scale
        )
        cand_h_list.append(h_sl)
        cand_nv_list.append(nv_sl)
        cand_ev_list.append(ev_sl + math.log(0.075))
        cand_cfo_list.append(f_c)
        cand_names.append(nm)
        raw_feats.append(ft_sl)
        family_ids.append(2)

    # Family 3: Piecewise mid-packet phase-jump branches (total prior 0.20 -> 0.10 each)
    for nm, f_c in (("pw_jump_local", cfo_local), ("pw_jump_ref", cfo_ref1)):
        h_pw, nv_pw, ev_pw, ft_pw = _fit_piecewise_jump_branch(
            y_sym, pkt.x_known, f_c, p_h_common, nv_common, 96, cfg.var_calib_scale
        )
        cand_h_list.append(h_pw)
        cand_nv_list.append(nv_pw)
        cand_ev_list.append(ev_pw + math.log(0.10))
        cand_cfo_list.append(f_c)
        cand_names.append(nm)
        raw_feats.append(ft_pw)
        family_ids.append(3)

    h_stack = np.stack(cand_h_list, axis=0)      # (12, 256)
    nv_stack = np.stack(cand_nv_list, axis=0)    # (12, 256)
    log_ev = np.array(cand_ev_list, dtype=np.float64)
    cfo_arr = np.array(cand_cfo_list, dtype=np.float64)

    # Build normalized, scale-invariant, strictly pilot-observable feature matrix (12, 12) for Exp C
    ev_rel = log_ev - float(np.max(log_ev))
    w_init = _softmax(log_ev, temp=1.0)
    h_bar = np.sum(w_init[:, None] * h_stack, axis=0)
    h_scale = math.sqrt(max(p_h_common, 1e-18))
    snr_db_est = 10.0 * math.log10(max(p_h_common / nv_common, 1e-6))

    feats_mat = np.zeros((len(cand_names), 12), dtype=np.float64)
    for r in range(len(cand_names)):
        ft = raw_feats[r]
        dist_to_mean = float(np.sqrt(np.mean(np.abs(h_stack[r] - h_bar) ** 2)) / (h_scale + 1e-18))
        fam_oh = [1.0 if family_ids[r] == f else 0.0 for f in range(4)]
        feats_mat[r] = [
            float(np.clip(ev_rel[r] / 10.0, -5.0, 0.0)),
            math.log1p(max(ft["res_sync"], 0.0)),
            math.log1p(max(ft["res_tail"], 0.0)),
            ft["ph_diff"] / math.pi,
            (cfo_arr[r] - cfo_ref1) / 500.0,
            (cfo_arr[r] - cfo_local) / 500.0,
            dist_to_mean,
            snr_db_est / 20.0,
            *fam_oh,
        ]

    return CandidateBank(
        y_sym=y_sym,
        timing_rx=timing_rx,
        h_cands=h_stack,
        nv_cands=nv_stack,
        log_ev=log_ev,
        cand_cfos=cfo_arr,
        cand_feats=feats_mat,
        cand_names=tuple(cand_names),
        build_ms=(time.perf_counter() - t0) * 1000.0,
    )


_PAYLOAD_MASK = np.zeros(PKT_LEN, dtype=np.float64)
_PAYLOAD_MASK[PAYLOAD_IDX] = 1.0


def decode_from_candidate_bank(
    bank: CandidateBank,
    weights: np.ndarray,
    *,
    mode: str,
    block_temp: float = 0.25,
    h_override: np.ndarray | None = None,
    nv_override: np.ndarray | None = None,
) -> ReceiverOutput:
    """Decode QPSK packet from a fixed candidate bank and posterior weights using specified fusion rule.

    Modes (paired on identical bank & pilot-only prior weights `weights`):
      - `hard_select`: MAP single candidate r* = argmax_r alpha_r
      - `linear_fusion_plugin`: posterior mean h_mix = sum_r alpha_r h_r + TOTAL variance
        (within + between, posterior_fusion §4.1), then Gaussian plug-in QPSK demodulation
      - `linear_within_only`: posterior mean h_mix with WITHIN variance only (tests §4.1 under-variance)
      - `sym_mixture`: per-symbol independent mixture likelihood sum_r alpha_r CN(y_n | h_r x, nv_r)
      - `mixture_likelihood`: EXACT block-joint LOO mixture posterior (posterior_fusion §5.2):
        updates branch posterior beta_r^{(-n)} using unlabeled constellation likelihood across the
        shared payload block, then marginalizes mixture into exact QPSK MAP & log-sum-exp bit LLR
      - `avg_llr_antipattern`: linear average of per-branch LLRs sum_r alpha_r LLR_r (negative control §5.2)
    """
    t0 = time.perf_counter()
    h_c = bank.h_cands if h_override is None else h_override
    nv_c = bank.nv_cands if nv_override is None else nv_override
    w = np.asarray(weights, dtype=np.float64)
    w = np.maximum(w, 0.0)
    w /= max(float(np.sum(w)), 1e-18)

    # Exact posterior mean & within/between covariance decomposition (posterior_fusion §4.1)
    h_mix = np.sum(w[:, None] * h_c, axis=0)
    v_within = np.sum(w[:, None] * nv_c, axis=0)
    v_between = np.sum(w[:, None] * (np.abs(h_c - h_mix[None, :]) ** 2), axis=0)
    v_total = np.maximum(v_within + v_between, 1e-18)

    r_hat = int(np.argmax(w))
    cfo_mix = float(np.sum(w * bank.cand_cfos))
    entropy = float(-np.sum(w * np.log(np.maximum(w, 1e-300))))

    if mode == "hard_select":
        h_out = h_c[r_hat]
        nv_out = nv_c[r_hat]
        x_eq = np.conj(h_out) * bank.y_sym / (np.abs(h_out) ** 2 + nv_out)
        bit_llrs = qpsk_bit_llrs_gaussian(bank.y_sym, h_out, nv_out)
        hard_syms = qpsk_hard_symbols(x_eq)
        est_cfo = float(bank.cand_cfos[r_hat])
    elif mode == "linear_fusion_plugin":
        h_out = h_mix
        nv_out = v_total
        x_eq = np.conj(h_out) * bank.y_sym / (np.abs(h_out) ** 2 + nv_out)
        bit_llrs = qpsk_bit_llrs_gaussian(bank.y_sym, h_out, nv_out)
        hard_syms = qpsk_hard_symbols(x_eq)
        est_cfo = cfo_mix
    elif mode == "linear_within_only":
        h_out = h_mix
        nv_out = np.maximum(v_within, 1e-18)
        x_eq = np.conj(h_out) * bank.y_sym / (np.abs(h_out) ** 2 + nv_out)
        bit_llrs = qpsk_bit_llrs_gaussian(bank.y_sym, h_out, nv_out)
        hard_syms = qpsk_hard_symbols(x_eq)
        est_cfo = cfo_mix
    elif mode == "sym_mixture":
        h_out = h_mix
        nv_out = v_total
        hard_syms, bit_llrs, sym_ll = mixture_qpsk_posterior(bank.y_sym, h_c, nv_c, w)
        sym_ll_s = sym_ll - np.max(sym_ll, axis=1, keepdims=True)
        sym_p = np.exp(sym_ll_s)
        sym_p /= np.sum(sym_p, axis=1, keepdims=True)
        x_eq = sym_p @ QPSK_CONSTELLATION
        est_cfo = cfo_mix
    elif mode == "mixture_likelihood":
        # Full block-joint LOO posterior mixture (strictly unlabeled payload constellation marginalization)
        hard_syms, bit_llrs, sym_ll, beta_full = joint_mixture_qpsk_posterior(
            bank.y_sym, h_c, nv_c, w, payload_mask=_PAYLOAD_MASK, block_temp=block_temp
        )
        # Keep holdout channel readout strictly pilot-only (h_mix, v_total) so holdout evaluation
        # never uses payload observations, OR note exact posterior readout! Using pilot-only (h_mix, v_total)
        # keeps channel estimation strictly comparable on identical pilot information I!
        h_out = h_mix
        nv_out = v_total
        sym_ll_s = sym_ll - np.max(sym_ll, axis=1, keepdims=True)
        sym_p = np.exp(sym_ll_s)
        sym_p /= np.sum(sym_p, axis=1, keepdims=True)
        x_eq = sym_p @ QPSK_CONSTELLATION
        est_cfo = float(np.sum(beta_full * bank.cand_cfos))
    elif mode == "avg_llr_antipattern":
        h_out = h_mix
        nv_out = v_total
        llrs_each = np.stack([qpsk_bit_llrs_gaussian(bank.y_sym, h_c[r], nv_c[r]) for r in range(len(w))], axis=0)
        bit_llrs = np.sum(w[:, None] * llrs_each, axis=0)
        dec_bits = (bit_llrs > 0.0).astype(np.uint8)
        hard_syms = bits_to_qpsk(dec_bits)
        x_eq = np.conj(h_out) * bank.y_sym / (np.abs(h_out) ** 2 + nv_out)
        est_cfo = cfo_mix
    else:
        raise ValueError(f"Unknown Experiment B decode mode: {mode}")

    dec_ms = (time.perf_counter() - t0) * 1000.0
    proj_gap = float(np.mean(np.abs(h_c[r_hat] - h_mix) ** 2))
    return ReceiverOutput(
        y_sym=bank.y_sym,
        h_hat=h_out,
        nv_hat=nv_out,
        x_eq=x_eq,
        bit_llrs=bit_llrs,
        hard_syms=hard_syms,
        est_timing_rx=bank.timing_rx,
        est_cfo_hz=est_cfo,
        est_noise_var=float(np.median(nv_out)),
        elapsed_ms=bank.build_ms + dec_ms,
        extra={
            "decode_only_ms": dec_ms,
            "bank_build_ms": bank.build_ms,
            "top_cand_idx": r_hat,
            "top_cand_weight": float(w[r_hat]),
            "weight_entropy": entropy,
            "proj_gap_mse": proj_gap,
            "between_var_mean": float(np.mean(v_between)),
            "within_var_mean": float(np.mean(v_within)),
        },
    )


def run_experiment_b_arms(
    bank: CandidateBank,
    *,
    post_temp: float = 1.5,
    block_temp: float = 0.25,
) -> dict[str, ReceiverOutput]:
    """Run all 6 Experiment B arms on the EXACT same CandidateBank and pilot posterior weights."""
    w = _softmax(bank.log_ev, temp=post_temp)
    return {
        "B1_hard_select": decode_from_candidate_bank(bank, w, mode="hard_select"),
        "B2_linear_fusion_plugin": decode_from_candidate_bank(bank, w, mode="linear_fusion_plugin"),
        "B2b_linear_within_only": decode_from_candidate_bank(bank, w, mode="linear_within_only"),
        "B3a_sym_mixture": decode_from_candidate_bank(bank, w, mode="sym_mixture"),
        "B3_mixture_likelihood": decode_from_candidate_bank(bank, w, mode="mixture_likelihood", block_temp=block_temp),
        "B4_avg_llr_antipattern": decode_from_candidate_bank(bank, w, mode="avg_llr_antipattern"),
    }


# =============================================================================
# Experiment C: Lightweight Trainable PyTorch Attention / Weight + Residual Net
# =============================================================================

def _require_torch():
    import torch
    torch.set_num_threads(1)
    return torch


def build_c_sample_arrays(
    pkt: PacketSample,
    bank: CandidateBank,
) -> dict[str, np.ndarray]:
    """Pack compact float32 tensors for fast single-threaded PyTorch training/inference on fixed CandidateBank."""
    # Scale normalize complex signals per packet by rms(|y_sym|) so neural inputs/losses are well-conditioned
    scale = float(np.sqrt(np.mean(np.abs(bank.y_sym) ** 2)) + 1e-18)
    y_pay = bank.y_sym[PAYLOAD_IDX] / scale
    h_pay = bank.h_cands[:, PAYLOAD_IDX] / scale
    nv_pay = bank.nv_cands[:, PAYLOAD_IDX] / (scale * scale)
    y_hold = bank.y_sym[HOLDOUT_IDX] / scale
    h_hold = bank.h_cands[:, HOLDOUT_IDX] / scale
    nv_hold = bank.nv_cands[:, HOLDOUT_IDX] / (scale * scale)
    x_hold = pkt.x_all_true[HOLDOUT_IDX]
    bits_pay = qpsk_to_bits(pkt.x_all_true[PAYLOAD_IDX]).astype(np.float32)

    return {
        "feats": bank.cand_feats.astype(np.float32),                 # (L, 12)
        "log_ev": (bank.log_ev - float(np.max(bank.log_ev))).astype(np.float32),  # (L,)
        "y_pay_re": y_pay.real.astype(np.float32),                   # (224,)
        "y_pay_im": y_pay.imag.astype(np.float32),
        "h_pay_re": h_pay.real.astype(np.float32),                   # (L, 224)
        "h_pay_im": h_pay.imag.astype(np.float32),
        "nv_pay": np.maximum(nv_pay, 1e-8).astype(np.float32),       # (L, 224)
        "bits_pay": bits_pay,                                        # (448,)
        "y_hold_re": y_hold.real.astype(np.float32),                 # (16,)
        "y_hold_im": y_hold.imag.astype(np.float32),
        "h_hold_re": h_hold.real.astype(np.float32),                 # (L, 16)
        "h_hold_im": h_hold.imag.astype(np.float32),
        "nv_hold": np.maximum(nv_hold, 1e-8).astype(np.float32),     # (L, 16)
        "x_hold_re": x_hold.real.astype(np.float32),
        "x_hold_im": x_hold.imag.astype(np.float32),
    }


def train_and_infer_experiment_c(
    train_pkts: Sequence[PacketSample],
    train_banks: Sequence[CandidateBank],
    dev_pkts: Sequence[PacketSample],
    dev_banks: Sequence[CandidateBank],
    test_pkts: Sequence[PacketSample],
    test_banks: Sequence[CandidateBank],
    *,
    init_temp: float = 1.5,
    block_temp: float = 0.25,
    seeds: Sequence[int] = (20261121, 20261122, 20261123),
    epochs: int = 45,
    batch_size: int = 64,
    lr: float = 4e-3,
) -> dict[str, Any]:
    """Train two lightweight single-thread PyTorch modules over the FIXED candidate bank:

      - C1 (`C1_attn_weight_net`): Cross-candidate self-attention weight calibration network
        predicting residual logits over fixed physical log-evidence (NO channel modification).
      - C2 (`C2_attn_weight_resid_net`): Same attention weight network + bounded per-candidate
        complex gain/phase residual & variance calibration head.

    Both feed the EXACT SAME fixed mixture-likelihood QPSK decoder (`decode_from_candidate_bank`).
    Strictly trained on `train`, model-selected & temperature-calibrated on `dev`, evaluated on `test`.
    """
    torch = _require_torch()
    import torch.nn as nn
    import torch.nn.functional as F

    class LightweightFusionNet(nn.Module):
        def __init__(self, in_dim: int = 12, d_model: int = 24, use_resid: bool = False, init_t: float = 1.5):
            super().__init__()
            self.use_resid = use_resid
            self.enc = nn.Sequential(
                nn.Linear(in_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            # Single-head cross-candidate scaled dot-product self-attention
            self.wq = nn.Linear(d_model, d_model, bias=False)
            self.wk = nn.Linear(d_model, d_model, bias=False)
            self.wv = nn.Linear(d_model, d_model, bias=False)
            self.norm = nn.LayerNorm(d_model)
            self.logit_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1),
            )
            # Learnable log inverse-temperature on physical evidence baseline
            self.log_inv_temp = nn.Parameter(torch.tensor(math.log(1.0 / max(init_t, 0.1)), dtype=torch.float32))
            if use_resid:
                # Bounded per-candidate (d_amp, d_phase, d_logvar) residual head initialized near zero
                self.resid_head = nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.GELU(),
                    nn.Linear(d_model // 2, 3),
                )
                nn.init.zeros_(self.resid_head[-1].weight)
                nn.init.zeros_(self.resid_head[-1].bias)
            nn.init.zeros_(self.logit_head[-1].weight)
            nn.init.zeros_(self.logit_head[-1].bias)

        def forward(self, feats: torch.Tensor, log_ev: torch.Tensor):
            # feats: (B, L, D), log_ev: (B, L)
            x = self.enc(feats)
            q = self.wq(x)
            k = self.wk(x)
            v = self.wv(x)
            attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
            attn_w = torch.softmax(attn_logits, dim=-1)
            h = self.norm(x + torch.matmul(attn_w, v))
            delta_s = self.logit_head(h).squeeze(-1)  # (B, L)
            inv_temp = torch.exp(self.log_inv_temp).clamp(0.05, 5.0)
            logits = inv_temp * log_ev + delta_s
            weights = torch.softmax(logits, dim=-1)
            if self.use_resid:
                r_raw = self.resid_head(h)  # (B, L, 3)
                d_amp = 0.15 * torch.tanh(r_raw[..., 0])
                d_ph = 0.18 * torch.tanh(r_raw[..., 1])
                d_lv = 0.60 * torch.tanh(r_raw[..., 2])
                return weights, d_amp, d_ph, d_lv
            return weights, None, None, None

    # Differentiable exact mixture QPSK bit LLR loss in PyTorch (matches evaluator.mixture_qpsk_posterior)
    c_re = torch.tensor([1.0, 1.0, -1.0, -1.0], dtype=torch.float32) / math.sqrt(2.0)
    c_im = torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=torch.float32) / math.sqrt(2.0)

    def batch_loss(model: LightweightFusionNet, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        w, d_amp, d_ph, d_lv = model(batch["feats"], batch["log_ev"])
        h_re = batch["h_pay_re"]
        h_im = batch["h_pay_im"]
        nv = batch["nv_pay"]
        if d_amp is not None:
            g_re = (1.0 + d_amp) * torch.cos(d_ph)
            g_im = (1.0 + d_amp) * torch.sin(d_ph)
            h_re_new = h_re * g_re.unsqueeze(-1) - h_im * g_im.unsqueeze(-1)
            h_im_new = h_re * g_im.unsqueeze(-1) + h_im * g_re.unsqueeze(-1)
            h_re, h_im = h_re_new, h_im_new
            nv = nv * torch.exp(d_lv).unsqueeze(-1)

        # (B, L, N, 4)
        diff_re = batch["y_pay_re"][:, None, :, None] - (
            h_re[..., None] * c_re[None, None, None, :] - h_im[..., None] * c_im[None, None, None, :]
        )
        diff_im = batch["y_pay_im"][:, None, :, None] - (
            h_re[..., None] * c_im[None, None, None, :] + h_im[..., None] * c_re[None, None, None, :]
        )
        log_g = -torch.log(math.pi * nv[..., None]) - (diff_re ** 2 + diff_im ** 2) / nv[..., None]
        # Exact differentiable leave-one-out block-joint posterior matching fixed backend decoder
        log_py_sym = torch.logsumexp(log_g, dim=-1) - math.log(4.0)  # (B, L, N)
        tot_pay_ll = torch.sum(log_py_sym, dim=-1, keepdim=True)     # (B, L, 1)
        loo_ll = tot_pay_ll - log_py_sym                             # (B, L, N)
        log_w = torch.log(w.clamp_min(1e-12))[:, :, None]            # (B, L, 1)
        log_beta_loo = torch.log_softmax(log_w + float(block_temp) * loo_ll, dim=1)  # (B, L, N)
        sym_ll = torch.logsumexp(log_beta_loo[..., None] + log_g, dim=1)  # (B, N, 4)

        ll_b0_0 = torch.logaddexp(sym_ll[..., 0], sym_ll[..., 1])
        ll_b0_1 = torch.logaddexp(sym_ll[..., 2], sym_ll[..., 3])
        ll_b1_0 = torch.logaddexp(sym_ll[..., 0], sym_ll[..., 2])
        ll_b1_1 = torch.logaddexp(sym_ll[..., 1], sym_ll[..., 3])
        llr0 = ll_b0_1 - ll_b0_0
        llr1 = ll_b1_1 - ll_b1_0
        llrs = torch.stack([llr0, llr1], dim=-1).reshape(sym_ll.shape[0], -1)
        bce = F.binary_cross_entropy_with_logits(llrs, batch["bits_pay"])

        # Auxiliary channel prediction MSE on training payload/holdout? Note: in train split, holdout pilots
        # are available for supervised channel calibration (never touched in dev/test!)
        if d_amp is not None:
            hh_re = batch["h_hold_re"] * g_re.unsqueeze(-1) - batch["h_hold_im"] * g_im.unsqueeze(-1)
            hh_im = batch["h_hold_re"] * g_im.unsqueeze(-1) + batch["h_hold_im"] * g_re.unsqueeze(-1)
        else:
            hh_re, hh_im = batch["h_hold_re"], batch["h_hold_im"]
        hm_re = torch.sum(w.unsqueeze(-1) * hh_re, dim=1)
        hm_im = torch.sum(w.unsqueeze(-1) * hh_im, dim=1)
        yh_re = hm_re * batch["x_hold_re"] - hm_im * batch["x_hold_im"]
        yh_im = hm_re * batch["x_hold_im"] + hm_im * batch["x_hold_re"]
        ch_mse = torch.mean((batch["y_hold_re"] - yh_re) ** 2 + (batch["y_hold_im"] - yh_im) ** 2)
        return bce + 0.10 * ch_mse

    def stack_split(pkts: Sequence[PacketSample], banks: Sequence[CandidateBank]) -> dict[str, torch.Tensor]:
        arrs = [build_c_sample_arrays(p, b) for p, b in zip(pkts, banks)]
        return {k: torch.from_numpy(np.stack([a[k] for a in arrs], axis=0)) for k in arrs[0].keys()}

    tr_data = stack_split(train_pkts, train_banks)
    dv_data = stack_split(dev_pkts, dev_banks)
    n_tr = len(train_pkts)

    results_c: dict[str, Any] = {}
    for variant_name, use_resid in (("C1_attn_weight_net", False), ("C2_attn_weight_resid_net", True)):
        seed_runs: list[dict[str, Any]] = []
        test_outputs_by_seed: dict[int, list[ReceiverOutput]] = {}
        dev_outputs_by_seed: dict[int, list[ReceiverOutput]] = {}

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            tracemalloc.start()
            t_tr0 = time.perf_counter()

            model = LightweightFusionNet(in_dim=12, d_model=24, use_resid=use_resid, init_t=init_temp)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-3)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

            best_dev_loss = float("inf")
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = 0

            gen = torch.Generator().manual_seed(seed)
            for ep in range(1, epochs + 1):
                model.train()
                perm = torch.randperm(n_tr, generator=gen)
                for i0 in range(0, n_tr, batch_size):
                    idx = perm[i0 : i0 + batch_size]
                    mb = {k: v[idx] for k, v in tr_data.items()}
                    opt.zero_grad()
                    loss = batch_loss(model, mb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    opt.step()
                sched.step()

                model.eval()
                with torch.no_grad():
                    dv_loss = float(batch_loss(model, dv_data).item())
                if dv_loss < best_dev_loss:
                    best_dev_loss = dv_loss
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    best_epoch = ep

            train_sec = time.perf_counter() - t_tr0
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            model.load_state_dict(best_state)
            model.eval()

            def _infer_split(banks: Sequence[CandidateBank]) -> list[ReceiverOutput]:
                outs: list[ReceiverOutput] = []
                for bk in banks:
                    t_i0 = time.perf_counter()
                    ft = torch.from_numpy(bk.cand_feats.astype(np.float32))[None, :, :]
                    lev = torch.from_numpy((bk.log_ev - float(np.max(bk.log_ev))).astype(np.float32))[None, :]
                    with torch.no_grad():
                        w_t, da_t, dp_t, dlv_t = model(ft, lev)
                    w_np = w_t[0].numpy().astype(np.float64)
                    if da_t is not None:
                        da = da_t[0].numpy().astype(np.float64)
                        dp = dp_t[0].numpy().astype(np.float64)
                        dlv = dlv_t[0].numpy().astype(np.float64)
                        gain = (1.0 + da) * np.exp(1j * dp)
                        h_mod = bk.h_cands * gain[:, None]
                        nv_mod = bk.nv_cands * np.exp(dlv)[:, None]
                    else:
                        h_mod, nv_mod = None, None
                    nn_ms = (time.perf_counter() - t_i0) * 1000.0
                    ro = decode_from_candidate_bank(
                        bk, w_np, mode="mixture_likelihood", block_temp=block_temp, h_override=h_mod, nv_override=nv_mod
                    )
                    ro.elapsed_ms += nn_ms
                    ro.extra["nn_infer_ms"] = nn_ms
                    ro.extra["seed"] = seed
                    outs.append(ro)
                return outs

            dev_outs = _infer_split(dev_banks)
            test_outs = _infer_split(test_banks)
            dev_outputs_by_seed[seed] = dev_outs
            test_outputs_by_seed[seed] = test_outs
            seed_runs.append(
                {
                    "seed": seed,
                    "n_params": n_params,
                    "best_epoch": best_epoch,
                    "best_dev_loss": best_dev_loss,
                    "train_time_sec": train_sec,
                    "train_peak_py_mem_mb": peak_bytes / (1024.0 * 1024.0),
                    "learned_inv_temp": float(torch.exp(model.log_inv_temp).item()),
                }
            )

        # Select primary seed by lowest dev loss (zero test peeking!) and also keep all seeds
        best_seed_rec = min(seed_runs, key=lambda r: r["best_dev_loss"])
        results_c[variant_name] = {
            "seed_runs": seed_runs,
            "dev_selected_seed": best_seed_rec["seed"],
            "dev_outputs_by_seed": dev_outputs_by_seed,
            "test_outputs_by_seed": test_outputs_by_seed,
        }
    return results_c
"""Download and verify CTTC Indoor Wireless Channel QPSK datasets (Zenodo 4895133).

Official source:
  Adriano Pastore & Armin Ghani (CTTC), "Datasets of Indoor Wireless Channel Measurements
  for Machine Learning Applications", Zenodo, June 2, 2021.
  DOI: 10.5281/zenodo.4895133 | URL: https://zenodo.org/records/4895133

Run on remote compute host (e.g. GitHub Codespace into /tmp/agy_cli1_cttc/raw).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
import urllib.request
from pathlib import Path
from typing import Any

ZENODO_RECORD_ID = "4895133"
ZENODO_BASE_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}/files"

MANIFEST_FILES: list[dict[str, Any]] = [
    {
        "filename": "README.pdf",
        "role": "documentation",
        "size_bytes": 1549089,
        "md5": "620c116c2a38a6dfbb6316752833850d",
        "sha256": "c8e7ae57e0896dfe2f1cc910d084e9b77e3cc5143b721cf9b24753ab8b496fb4",
    },
    {
        "filename": "sources.tar.gz",
        "role": "tx_source_bits",
        "size_bytes": 157051651,
        "md5": "48766e3f5ddaa968ccf128eb5891a450",
        "sha256": "f9022be37d3eeadcff481f46b25ca9a067c0e19988bc4bdcb77cca9957ee7b4f",
        "extracted_files": {
            "source_QPSK.dat": {
                "size_bytes": 1000000,
                "sha256": "6dc9635b72b34f2d792f7ec6522a0450181397fb616262b629ee593e35f3f1cb",
            }
        },
    },
    {
        "filename": "f2450_snr20_QPSK.tar.gz",
        "role": "capture",
        "capture_id": "f2450_snr20_QPSK",
        "freq_mhz": 2450,
        "nom_snr_db": 20,
        "size_bytes": 211697276,
        "md5": "1df69db08519d130faff0857347930de",
        "sha256": "161d618ef2da2412a3c4d7236deb7c52f0b81f31af704b9d852d90f849c6cbb3",
        "extracted_files": {
            "config.txt": {"size_bytes": 287, "sha256": "7e9e6ff2af0f10f2cf0a47679394a6443a1fb8051c6e0c61acacd29c55a5389e"},
            "file_1.dat": {"size_bytes": 176015264, "sha256": "c6596f98fd9705c475b8b07340facb6ad7dbc4906e6f196ac2f207a275544ab1"},
            "file_2.dat": {"size_bytes": 176015264, "sha256": "20c521344958acda0cd30dcd0449a41ab16d472e6942bee0d5e539c788c7a4db"},
        },
    },
    {
        "filename": "f2450_snr10_QPSK.tar.gz",
        "role": "capture",
        "capture_id": "f2450_snr10_QPSK",
        "freq_mhz": 2450,
        "nom_snr_db": 10,
        "size_bytes": 197835355,
        "md5": "545e2f96b666200da371bb51726bb336",
        "sha256": "e6b7462ec66d485b1b10936e3f86601c91e900dddad9b8b9e19b789c3e183d25",
        "extracted_files": {
            "config.txt": {"size_bytes": 287, "sha256": "b9dde9083a6b228929913b577f2bfc2e2e9e50a61638a0dbe68bc7fccf6cd221"},
            "file_1.dat": {"size_bytes": 176015264, "sha256": "3d56b0f14b97755ecdfae2011f011165a6837c2caa363a9cbfaedc0dfcd53d0c"},
            "file_2.dat": {"size_bytes": 176015264, "sha256": "b91aba4447771e7dfd149515067ee64e1303ae44c6669d65504f17899070f331"},
        },
    },
    {
        "filename": "f2450_snr0_QPSK.tar.gz",
        "role": "capture",
        "capture_id": "f2450_snr0_QPSK",
        "freq_mhz": 2450,
        "nom_snr_db": 0,
        "size_bytes": 191552426,
        "md5": "8ef953dfadfb7363d70924f79f4755bc",
        "sha256": "e62681b28bb02c51671ffe3f3c36030d7a0e0839440adbf641a8e63f53b95ea9",
        "extracted_files": {
            "config.txt": {"size_bytes": 287, "sha256": "111e8a12b35d5529e8e49c781de80d2ad05172e73ef2e65f747229ad0f63bf07"},
            "file_1.dat": {"size_bytes": 176015264, "sha256": "14cdb93e54351c5d9b2ac254a4f670863dd4a83bfa7ed48695186d64bf2460d2"},
            "file_2.dat": {"size_bytes": 176015264, "sha256": "7c390cdf2a55f800c2a1ad46e0cdf732630cc9cc3a48ab763d6f39184c3ff615"},
        },
    },
    {
        "filename": "f708_snr20_QPSK.tar.gz",
        "role": "capture",
        "capture_id": "f708_snr20_QPSK",
        "freq_mhz": 708,
        "nom_snr_db": 20,
        "size_bytes": 212954092,
        "md5": "afbd783156079f3ab2b1a33690540bad",
        "sha256": "1732da579f2e98ade65e2570f4141e9b881d6c891f8dec83a3bd723e05f8d1af",
        "extracted_files": {
            "config.txt": {"size_bytes": 285, "sha256": "a4569e6bce65a47dc227470f36b7d1607db32a1049011bda3440d296cc1b21e5"},
            "file_1.dat": {"size_bytes": 176015264, "sha256": "1f2b2b22c2345df8b2e67f0a7a44e525a65615723cf878eb63f231dc302e2436"},
            "file_2.dat": {"size_bytes": 176015264, "sha256": "8d05494ec48bae4456f4f96cbfb3a648fc81c9bf161388432e82d05f89b8a361"},
        },
    },
    {
        "filename": "f708_snr10_QPSK.tar.gz",
        "role": "capture",
        "capture_id": "f708_snr10_QPSK",
        "freq_mhz": 708,
        "nom_snr_db": 10,
        "size_bytes": 198519362,
        "md5": "fe9ceeb85392f1476289214736640b1c",
        "sha256": "b688f19db545edeef24af628283ff5080b7bd32f0277fd593b44a24da5a791f9",
        "extracted_files": {
            "config.txt": {"size_bytes": 285, "sha256": "e9133b9cf31bcee6c36eb5f10e4da78cf60129f2fce22de1f0d9e4cd3917b0c3"},
            "file_1.dat": {"size_bytes": 176015264, "sha256": "962c8e5a41d8d4b6827dc8c6ef09bb7a931d1ae4c19c89203794bf1b482cb21e"},
            "file_2.dat": {"size_bytes": 176015264, "sha256": "6a7cd6c531d9125a2baa66c391de06d83d1079deebb4cf240a6bf2755051c1d9"},
        },
    },
    {
        "filename": "f708_snr0_QPSK.tar.gz",
        "role": "capture",
        "capture_id": "f708_snr0_QPSK",
        "freq_mhz": 708,
        "nom_snr_db": 0,
        "size_bytes": 190371348,
        "md5": "d971092391ddccc893791ece9ad71bf3",
        "sha256": "adc850b02149e0cf9a6f27bdba2c93a3290fdf16d2d414725dbcaccca7a22a00",
        "extracted_files": {
            "config.txt": {"size_bytes": 285, "sha256": "4312bb95378d0d9c4fe67f57267a256de9834459a59e6d636e58edcd8ab52a6d"},
            "file_1.dat": {"size_bytes": 176015264, "sha256": "6f67ca836a315aafa5e00328a54fae5a0e350dac96d4e9070012205fb24062dd"},
            "file_2.dat": {"size_bytes": 176015264, "sha256": "1b7815e226e3ececc50f9a9027a5e15f3fbd09514bb86889b912ba51596c4b85"},
        },
    },
    {
        "filename": "f433_snr20_QPSK.tar.gz",
        "role": "capture",
        "capture_id": "f433_snr20_QPSK",
        "freq_mhz": 433,
        "nom_snr_db": 20,
        "size_bytes": 211968737,
        "md5": "8dbb0ce9f09b683de8dbc9dc76997b74",
        "sha256": "c6e28fae2d06e1038ae990fdb231b39b7cd37322fe5eb498a1f755766abcd512",
        "extracted_files": {
            "config.txt": {"size_bytes": 285, "sha256": "8c9a105246bcb27949fd1717029edaa1079264868316248fbab8825fa1087ab2"},
            "file_1.dat": {"size_bytes": 176015264, "sha256": "058e3fa662941648faac9512e2b0695a81fd5d9cdcab62bfb6c245d394c4a01c"},
            "file_2.dat": {"size_bytes": 176015264, "sha256": "379753c9b2f1d151d3228b7d8661a9d79e2ac182116ceb22f5b26275483b3e17"},
        },
    },
    {
        "filename": "f433_snr10_QPSK.tar.gz",
        "role": "capture",
        "capture_id": "f433_snr10_QPSK",
        "freq_mhz": 433,
        "nom_snr_db": 10,
        "size_bytes": 199205337,
        "md5": "8e8fa8e8822eab62320a0baebb87e7c5",
        "sha256": "2067a4fd82671d0b86fd47e8f3fab5f87573e8653c7dbc6321dcba23a701b5d4",
        "extracted_files": {
            "config.txt": {"size_bytes": 285, "sha256": "a68edb1cd13814063a1c2036722cc57209f60868a4517dfe884eefd31cf51c42"},
            "file_1.dat": {"size_bytes": 176015264, "sha256": "d635beed43c87efd7cbf6988157296d8259c556c34c6c3594d345eec539bf085"},
            "file_2.dat": {"size_bytes": 176015264, "sha256": "103d64391bd4dd85ac249bc6f75a5503c92a365e99e7d43bbe4a7d8774786b00"},
        },
    },
    {
        "filename": "f433_snr0_QPSK.tar.gz",
        "role": "capture",
        "capture_id": "f433_snr0_QPSK",
        "freq_mhz": 433,
        "nom_snr_db": 0,
        "size_bytes": 193721020,
        "md5": "9cc2a791ef1b07977b5c43760a06bda8",
        "sha256": "c7b7bcba97151ad4dadebe3c21fb3503d44126c6912e56e024a19d303b17ef2f",
        "extracted_files": {
            "config.txt": {"size_bytes": 285, "sha256": "8f9b54ba0153ae3f9e53a6ba6e231708934c7a1b42957120ee122ebe5ec9aeca"},
            "file_1.dat": {"size_bytes": 176015264, "sha256": "84f47c34c7daac92412742af7133ac984b2e713408b8b6e57665708c07a64e43"},
            "file_2.dat": {"size_bytes": 176015264, "sha256": "98d19a36fbc990d9fd7214eb7855482703f76650b5fca7ef1d513730e9541d54"},
        },
    },
]


def file_hashes(path: Path) -> tuple[int, str, str]:
    h_md5 = hashlib.md5()
    h_sha256 = hashlib.sha256()
    total = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            total += len(chunk)
            h_md5.update(chunk)
            h_sha256.update(chunk)
    return total, h_md5.hexdigest(), h_sha256.hexdigest()


def download_and_verify(raw_dir: Path, work_dir: Path, extract: bool = True) -> dict[str, Any]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    src_dir = work_dir / "sources"
    data_dir = work_dir / "data"
    if extract:
        src_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)

    report = {"zenodo_record": ZENODO_RECORD_ID, "doi": f"10.5281/zenodo.{ZENODO_RECORD_ID}", "files": []}
    for item in MANIFEST_FILES:
        fn = item["filename"]
        dst = raw_dir / fn
        url = f"{ZENODO_BASE_URL}/{fn}/content"
        if not dst.exists():
            print(f"[download] {fn} <- {url}")
            urllib.request.urlretrieve(url, dst)
        sz, md5, sha256 = file_hashes(dst)
        assert sz == item["size_bytes"], f"Size mismatch for {fn}: {sz} != {item['size_bytes']}"
        assert md5 == item["md5"], f"MD5 mismatch for {fn}: {md5} != {item['md5']}"
        assert sha256 == item["sha256"], f"SHA256 mismatch for {fn}: {sha256} != {item['sha256']}"
        print(f"[verified] {fn} size={sz} sha256={sha256[:16]}...")

        if extract and fn.endswith(".tar.gz"):
            target = src_dir if item["role"] == "tx_source_bits" else data_dir
            with tarfile.open(dst, "r:gz") as tf:
                tf.extractall(target)
            for rel, meta in item.get("extracted_files", {}).items():
                ep = (src_dir / rel) if item["role"] == "tx_source_bits" else (data_dir / item["capture_id"] / rel)
                esz, _, esha = file_hashes(ep)
                assert esz == meta["size_bytes"], f"Extracted size mismatch: {ep}"
                assert esha == meta["sha256"], f"Extracted sha256 mismatch: {ep}"
        report["files"].append(item)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify CTTC QPSK dataset archives.")
    parser.add_argument("--raw-dir", type=Path, default=Path("/tmp/agy_cli1_cttc/raw"))
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/agy_cli1_cttc/work"))
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument("--write-manifest", type=Path, default=None)
    args = parser.parse_args()
    rep = download_and_verify(args.raw_dir, args.work_dir, extract=not args.no_extract)
    if args.write_manifest is not None:
        args.write_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.write_manifest.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
```

### Event #8: `item.completed` — `command_execution` (`item_4`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command 'foreach ($file in '"'PROTOCOL.md','prepare_benchmarks.py','waveform_frontend.py','receivers.py','download_data.py') { git show \"origin/research/realdata-pluto-qpsk:research/realdata_pluto_qpsk/"'$file" }'
```

**Aggregated Output:**

```text
# EM-FloatFormer ADALM-PLUTO SDR QPSK 连续帧实测研究协议（PROTOCOL）

- **研究分支**：`research/realdata-pluto-qpsk`
- **起始基线提交**：`f566f9b8b0ddfb468a4cb3fd64588bfa46c87212`
- **远端执行环境**：GitHub Codespace（单线程限流 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCH_NUM_THREADS=1`，峰值内存严格 $< 2.5\text{ GB}$）
- **数据来源**：*Frames of SDR ADALM PLUTO based QPSK Communication Link*（Latha V, Remya Krishna, Pallaviram Sure，Mendeley Data, 2026-06-03），DOI `10.17632/k72b4pnc5g.1`，官方链接 `https://data.mendeley.com/datasets/k72b4pnc5g/1`
- **数据校验指纹**：`QPSKlink.mat`，`822,762,822` bytes，SHA-256 `c029532908f9702fea4be3ce37d0e1de9138019e12dd06e228abb1b83525f2b7`（与 Mendeley Public API 完全一致）

---

## 1. 实测波形物理结构与第一性原理参数确认

通过对 `QPSKlink.mat` 全部 `7,096` 帧的离线结构审计（`inspect_mat.py` 与 `audit_full_7096.py`），确证以下物理参数：

| 物理参数 | 实测确认值 | 验证依据 |
| :--- | :--- | :--- |
| 总帧数 $N$ | `7,096` 帧 | `Tx_data (126, 7096)`, `Rx_data (7560, 7096)`, 零 NaN/Inf |
| 载波频段与硬件链路 | 850 MHz，ADALM-PLUTO 单板收发（共享本振晶振），TX 0 dB / RX 20 dB，室内 3 m 多机架环境 | 官方元数据 + 极低残余 CFO（中位数 $6.0\times 10^{-6}\text{ rad/sym}$） |
| 过采样率 SPS | `20` samples/symbol | 周期自相关精确峰值在 lag = `2520 = 126 * 20`（中位 $|\rho| = 0.99955$） |
| 捕获窗口长度 | `7,560 = 3 * 2,520` samples/frame | 任意异步起始相位下均保证完整包含至少 2 个周期副本（Copy 1 & Copy 2） |
| 调制与星座映射 $\mathcal{Q}$ | 标准 QPSK：$\{(\pm 1 \pm j)\cdot 0.707\}$（$E_s = 0.999698$） | 全集唯一值遍历精确 4 点 |
| 比特映射（Gray 约定） | $b_0 = \mathbf{1}[\Re(s) < 0]$，$b_1 = \mathbf{1}[\Im(s) < 0]$ | $(+,+)\to 00, (+,-)\to 01, (-,+)\to 10, (-,-)\to 11$，单相邻错误恰差 1 bit |
| 帧内符号分区（126 符号） | **同步头 `[0:13]`**（Barker-13 副本 1）；**在线导频 `[13:21]`**（8 符号）；**留出参考探针 `[21:26]`**（5 符号，绝不进入估计）；**随机载荷 `[26:126]`**（100 符号 = 200 bits） | 前 26 符号全集恒定为 $(0.707+0.707j)\cdot[b_{13}, b_{13}]$；后 100 符号随机均匀 |
| 硬件零中频 AC 耦合效应 | $y_t[k] = h_t(s_t[k] - \bar{s}_t) + \text{ISI}_t[k] + w_t[k]$，即显著直流偏移 $d_t \approx -h_t \bar{s}_t$ | 实测 $d_t/h_t$ 与 $-\bar{s}_t$ 实/虚部相关系数高达 $r = 0.9989 / 0.9993$ |
| 信道时变与非平稳特征 | $|h_t| \in [0.0091, 0.0308]$（$>10.6\text{ dB}$ 衰落）；$\angle h_t \in [-0.61, +0.61]\text{ rad}$；在 $t \equiv 0 \pmod{20}$ 边界存在周期性批次跳变（p90 相对跳变 $36.4\%$，最大 $1.22\times$ / $0.94\text{ rad}$）；远端历史库（lag $> 50$）中 $76.4\%$ 的帧能找到比前一帧更近的复现状态 | 全 7,096 帧盲同步（100.0% 锁帧率）后的信道时序与重现审计 |

---

## 2. 防时间相关泄漏的连续大块划分与 `G = 16` 统计分组

由于真实室内无线信道存在强时间自相关（lag = 50 复相关性仍为 `0.937`），且存在每 `20` 帧一个硬件采集批次的跳变周期，严禁随机打乱样本。全数据集按时间严格顺序切分，所有切点对齐 `20`-帧边界，块间插入 **`100` 帧保护隔离带（Guard Band）**，且每个切分均配备**独立的 `200` 帧因果冷启动历史库**（测试集严禁使用训练/验证集的历史信道库，完全模拟独立在线流式部署）：

| 划分 | 原始帧索引区间 | 帧数 | 角色说明 |
| :--- | :--- | ---: | :--- |
| `train_warmup` | `[0, 200)` | 200 | 训练集因果历史库冷启动（不计损失） |
| **`train`** | **`[200, 3200)`** | **3,000** | 模型参数训练（150 个完整 20-帧批次） |
| `guard_1` | `[3200, 3300)` | 100 | 隔离保护带（丢弃，防止相邻帧信道泄漏） |
| `dev_warmup` | `[3300, 3500)` | 200 | 验证集独立因果冷启动（前 64 帧亦用于波形前端验证） |
| **`dev`** | **`[3500, 4500)`** | **1,000** | 经典超参调优、温度/方差校准与神经网络选模（`G=10` 簇 × 100 帧） |
| `guard_2` | `[4500, 4600)` | 100 | 隔离保护带（丢弃） |
| `test_warmup` | `[4600, 4800)` | 200 | 测试集独立因果冷启动（零跨集历史泄漏） |
| **`test`** | **`[4800, 7040)`** | **2,240** | **闭卷盲测正式评估集**（`G=16` 个连续时间簇 × 140 帧 = `224,000` 载荷符号 = `448,000` 比特） |
| `tail_diag` | `[7040, 7096)` | 56 | 尾段保留诊断（未满整簇，不计入主统计） |

---

## 3. 统一符号前端、因果历史库与四组对照设计

### 3.1 真实波形前端（Waveform Frontend）
输入为原始 `7,560`-sample 复基带波形（原始或加扰后）：
1. **粗 CFO 捕获**：结合 13-符号 Barker 差分相干（抗大频偏绕卷，$|\Delta\omega| \le 0.24\text{ rad/sym}$）与 lag-2520 周期相干精估；
2. **帧同步与亚采样定时恢复**：利用已知同步头 `[0:13]`（13 符号）在全部 20 个过采样相位与 `[0, 2520)` 起点窗口搜索归一化相干相关峰，并在峰值处执行 3 点抛物线亚采样偏移估计与 4 点三次（Catmull-Rom）插值重采样，抽取符号率主帧 $y_t[0:126]$（Copy 1）与独立副本 $y_t^{(2)}[0:126]$（Copy 2）。
3. **严格信息防火墙**：
   - 在线信道/DC/CFO 估计**仅允许读取**：当前帧在线导频 `[13:21]`（以及可选用于对比的全头 `[0:26]` 经典参考上界）、当前帧同步头 `[0:13]` 的盲能量/相关性摘要、以及严格因果历史帧库 $\mathcal{M}_t = \{t - L_{\text{lib}}, \dots, t - 1\}$（$L_{\text{lib}} = 200$）。
   - **历史帧库条目构造（100% 无真值泄漏）**：历史帧 $j < t$ 在自身完成解调后，仅用其自身的软后验符号期望 $\hat{s}_j^{\text{soft}}$（仅吸收高置信符号）联合重新估计并存入高精度复状态 $V_j = (\hat{h}_j^{\text{dd}}, \hat{\sigma}_{h,j}^2, \hat{d}_j^{\text{dd}}, \hat{\omega}_j^{\text{dd}}, q_j^{\text{rel}})$，**绝不读取离线 `Tx_data`**。
   - **留出探针 `[21:26]`**（5 符号）与**载荷真值 `[26:126]`**（100 符号）严格隔离，仅在离线评估器中可见。

### 3.2 成熟经典基线（Baseline）
- `base_ls_1tap`：单帧在线导频 `[13:21]` 朴素 1-tap LS 估计 + QPSK 相干解调；
- `base_ls_dc`：单帧在线导频 `[13:21]` 联合 $(h, d)$ 最小二乘（显式建模 ADALM-PLUTO 硬件直流消除）；
- `base_mmse_dc_cfo`（**核心经典单帧强基线**）：在线导频 `[13:21]` 联合 $(h, d)$ 经验贝叶斯 LMMSE 收缩 + 残余 CFO 细搜索 + 方差估计（正则超参仅在 `dev` 调优）；
- `base_full26_ls_dc`（参考上限对照）：使用全部 26 符号 `[0:26]` 联合 $(h, d)$ 估计的经典参考。

### 3.3 对照 A：全局参考范围与内容寻址（Content Addressing）
严格检验 `research/content_addressing_scalar` 定理：在同一因果历史库（$L_{\text{lib}} = 200$）与同一单帧导频观测下配对比较：
- `A1_local_single`：仅单帧导频 LMMSE（参考范围 $M=0$）；
- `A2_neighbor_k5`：固定近邻窗口（最近 $k=5$ 帧因果指数加权平均）作为先验与当前导频融合；
- `A3_global_uniform`：全局历史库（$M=200$）均匀平均先验与当前导频融合（隔离“扩大参考范围”的纯效应）；
- `A4_global_reliability`：全局历史库（$M=200$）按历史帧自身可靠性 $1/\hat{\sigma}_{h,j}^2$ 加权（不看当前帧内容匹配度）；
- `A5_global_content_attn`（**EM-FloatFormer 经典物理内容寻址**）：利用当前帧在线导频观测对历史库全部 $M=200$ 帧计算归一化物理匹配负对数似然分数 $s_{t,j}$，叠加显式 `self/clear` 重置保护分支，经 dev 校准温度 $\tau$ 做 softmax 内容寻址加权融合！并按 `t % 20 == 0`（真实硬件跳变边界）与 `t % 20 != 0`（段内平稳）分层解剖。

### 3.4 对照 B：硬选 vs 后验均值融合 vs 完整混合似然判决（Posterior Fusion）
严格检验 `research/posterior_fusion` 定理：构造**完全共享的离散物理候选库** $\mathcal{B}_t$：
- 维度 1：4 重 $\pi/2$ 相位旋转模糊分支 $q \in \{0, 1, 2, 3\}$；
- 维度 2：7 点对称细 CFO 漂移网格 $\omega_g \in \{-0.030, -0.020, -0.010, 0.0, +0.010, +0.020, +0.030\}\text{ rad/sym}$（围绕前端粗估残差中心化，兼顾常规与大 CFO 压力工况）；
- 维度 3：3 个物理信道轨迹先验族 $c \in \{\text{self-clear}, \text{local-neighbor}, \text{content-global}\}$；
- 共 $R = 4 \times 7 \times 3 = 84$ 个互斥物理分支，共享完全相同的前置后验权重 $\alpha_r = P(r \mid \mathcal{I}_t)$ 与条件高斯预测 $(\mu_{r,k}, d_r, v_{r,k})$。严格配对比较：
  - `B1_hard_map`：硬选单分支 $\hat{r} = \arg\max_r \alpha_r$ 做信道预测与判决；
  - `B2_posterior_mean_plugin`：融合后验均值 $\bar{h}_k = \sum_r \alpha_r h_{r,k}$ 与全协方差（分支内 + 分支间总方差）后做单高斯 plug-in 判决；
  - `B3a_sym_mixture`：保留完整 $R=84$ 分支混合分布，逐符号精确边缘化 QPSK 符号与比特似然（不跨载荷更新 $\alpha_r$）；
  - `B3b_joint_mixture`：按 `posterior_fusion` §5.2 完整公式，先用全帧载荷 QPSK 星座边际似然精确更新帧级分支后验 $\beta_r = P(r \mid \mathcal{I}_t, y_{\text{pay}})$，再边缘化全部 $R=84$ 分支进行精确符号 MAP 与逐比特 LLR MAP 判决！

### 3.5 对照 C：轻量级可训练 Attention / 候选权重校准与残差跟踪器（PyTorch CPU）
在完全相同的候选库与后端混合判决下，在 `train`（3,000 帧）上以 PyTorch CPU 单线程训练轻量级神经模块 `EMFloatFormerLite`（3 个独立随机种子：`20261121, 20261122, 20261123`）：
- 模块 1（**可学习内容寻址注意力**）：学习 query/key 投影与 `clear` 门控温度，从 $M=200$ 因果历史库生成自适应内容先验；
- 模块 2（**候选权重校准与残差精修头**）：预测共享候选库 $\mathcal{B}_t$ 的温度/先验对数偏置校正与复信道/残余频偏小幅残差修正，以及校准后的对数方差；
- 仅在 `dev` 上选模与方差温度校准，在 `test`（2,240 帧）上分别汇报各随机种子与均值±标准差，并与 dev 调优的强经典全局方法（`A5`、`B3b`）进行 `G=16` 配对检验。

---

## 4. 可控物理波形扰动压力测试（Stress Tests）

全部扰动以固定确定性种子直接叠加在原始 `7,560`-sample 实测复基带波形上，并经真实波形前端处理：

| 工况标识 | 物理扰动定义 | 核心检验目的 |
| :--- | :--- | :--- |
| `raw_original` | 零人工扰动（纯 ADALM-PLUTO 实测波形） | 真实室内衰落、硬件 DC 耦合、ISI 与 20-帧批次跳变 |
| `awgn_snr10db` | 叠加校准 AWGN 使目标符号有效 SNR = `10 dB` | 中信噪比下跨帧内容寻址增益 |
| `awgn_snr6db` | 叠加校准 AWGN 使目标符号有效 SNR = `6 dB` | 导频噪声显著时单帧 vs 近邻 vs 全局内容寻址分离 |
| `awgn_snr2db` | 叠加校准 AWGN 使目标符号有效 SNR = `2 dB` | 低 SNR 下 4 重相位模糊与后验融合增益 |
| `awgn_snr_m2db` | 叠加校准 AWGN 使目标符号有效 SNR = `-2 dB` | 极低 SNR 失效边界与不确定性校准 |
| `cfo_0p015_raw` | 纯实测波形叠加连续 CFO 漂移 $\Delta\omega \sim \mathcal{U}(\pm [0.012, 0.018])\text{ rad/sym}$ | 单帧累计 $>90^\circ$ 频偏漂移下 CFO 候选网格与混合后验消歧 |
| `cfo_0p020_snr6db` | 联合 CFO 漂移 $\Delta\omega \sim \mathcal{U}(\pm [0.015, 0.025])\text{ rad/sym}$ + AWGN `6 dB` | 频偏漂移与低 SNR 耦合的硬核 SDR 场景 |
| `phase_jump_snr6db` | 帧间 $25\%$ 概率发生离散象限/大角度突发相位跳变（$\Delta\phi \in \{\pm\pi/2, \pi, \pm\pi/3\}$）+ 帧内随机游走相位噪声（$\sigma_\phi = 0.03\text{ rad/sym}$）+ AWGN `6 dB` | 检验近邻滞后崩溃、内容寻址抗跳变、与 4 重模糊混合译码优势 |

---

## 5. 统一评估指标与统计检定口径（`evaluator.py`）

- **分区指标**：分别针对**同步头 `[0:13]`**、**在线导频 `[13:21]`**、**留出参考探针 `[21:26]`** 与**载荷 `[26:126]`** 报告：
  1. `SER`（Symbol Error Rate）：符号硬判决错率；
  2. `sym_rmse` / `sym_nmse`：均衡重建符号相对真值发送符号的均方根误差与归一化均方误差；
  3. `EVM (%)`：$\sqrt{\mathbb{E}| \hat{s} - s |^2 / \mathbb{E}|s|^2} \times 100\%$；
  4. `BER`（Bit Error Rate）：基于上述明确 Gray QPSK 位映射（2 bit/symbol）的错比特率；
  5. `FER`（Frame Error Rate）：定义为一帧 100 个载荷符号中存在 $\ge 1$ 个符号判决错误即记为错帧，分母为该评测集的全部有效帧数（同时另报端到端含失步帧的总分母 FER）；
  6. **信道与留出参考预测质量**：
     - 复信道 NMSE（相对离线 126-sym Oracle 信道 $h_t^{\text{ora}}$）；
     - 留出参考探针 `[21:26]` 预测 NMSE：$\sum |y_k - (\hat{h}_k s_k + \hat{d})|^2 / \sum |y_k|^2$；
     - 不确定性校准：留出探针复高斯平均负对数似然（`probe_nll`）、标准化残差均方（`z2_mean`，理想校准 = `1.0`）、以及理论 90% / 95% 复圆置信区域经验覆盖率（`cov90`, `cov95`，理想值 `0.90 / 0.95`）；
  7. **工程与覆盖率统计**：前端同步成功率、失步率、数值异常（NaN/Inf）率、单线程中位/平均推理耗时（ms/frame）、峰值 RSS 内存（MB）、候选分支数、训练总耗时与参数量；
  8. **严格分组统计推断（`test` 集 `G=16` 连续时间簇）**：配对簇差均值、95% Student-$t$（$df=15$）置信区间、簇级胜/平/负计数、精确 $2^{16} = 65,536$ 态配对符号翻转双侧 $p$ 值及 Holm-Bonferroni 族校正 $p_{\text{Holm}}$。
#!/usr/bin/env python3
"""Prepare real-waveform benchmark splits across raw_original and all 7 controllable stress conditions.

Directly injects physical perturbations onto the raw 7,560-sample ADALM-PLUTO complex waveforms
(from /tmp/agy_cli2_pluto/work/rx_c64.npy), executes the blind waveform receiver frontend
(matched correlation, timing recovery, fractional Catmull-Rom cubic resampling, frame alignment),
and saves compact symbol-rate benchmark bundles + frontend sync statistics for:
  - train (warmup [0:200] + train [200:3200], raw + multi-condition training pool)
  - dev   (warmup [3300:3500] + eval [3500:4500], all 8 conditions)
  - test  (warmup [4600:4800] + eval [4800:7040], all 8 conditions)

Strictly single-threaded and memory-safe (<500 MB peak RSS).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import time
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from waveform_frontend import (
    FRAME_SAMP,
    FULL_HDR_26,
    N_SYM,
    RX_SAMP,
    SPS,
    STRESS_SPECS,
    PerturbationSpec,
    cubic_catmull_rom_sample,
    inject_waveform_perturbation,
)

# Frozen non-overlapping temporal splits aligned to 20-frame hardware bursts with 100-frame guard bands
SPLITS = {
    "train": {"warmup_start": 0, "eval_start": 200, "eval_end": 3200},
    "dev": {"warmup_start": 3300, "eval_start": 3500, "eval_end": 4500},
    "test": {"warmup_start": 4600, "eval_start": 4800, "eval_end": 7040},
}

_HDR21 = FULL_HDR_26[:21]
_HDR21_CONJ = np.conj(_HDR21)
_HDR21_EN = float(np.sum(np.abs(_HDR21) ** 2))
_N_IDX = np.arange(RX_SAMP, dtype=np.float64)


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def fast_blind_frontend(
    r_wav: np.ndarray,
    single_shot_cfo_mode: bool = False,
) -> dict[str, Any]:
    """Fast vectorized blind waveform frontend (exact match to waveform_frontend.blind_waveform_frontend).

    When `single_shot_cfo_mode=True` (used for residual-CFO stress tests to model single-frame
    transmission without a repeated second copy for lag-2520 CFO cancellation), the frontend
    performs blind frame/timing synchronization using non-coherent segmented header correlation
    and leaves the intra-frame residual CFO intact for downstream carrier tracking / B-group mixture!
    """
    t0 = time.perf_counter()
    r = np.asarray(r_wav, dtype=np.complex128)

    if not single_shot_cfo_mode:
        # Standard periodic lag-2520 coarse CFO + two-copy coherent 21-sym header sync
        a = r[:-FRAME_SAMP]
        b = r[FRAME_SAMP:]
        c_per = np.vdot(a, b)
        en_a = float(np.sum(a.real ** 2 + a.imag ** 2))
        en_b = float(np.sum(b.real ** 2 + b.imag ** 2))
        rho_per = float(np.abs(c_per) / np.sqrt(en_a * en_b + 1e-30))
        omega_samp = float(np.angle(c_per)) / float(FRAME_SAMP)
        r_work = r * np.exp(-1j * omega_samp * _N_IDX)
        cfo_est = omega_samp * SPS

        r_dec = r_work.reshape(378, SPS).T  # (20, 378)
        sw1 = sliding_window_view(r_dec[:, :126 + 20], 21, axis=1)
        sw2 = sliding_window_view(r_dec[:, 126:252 + 20], 21, axis=1)
        ip1 = sw1 @ _HDR21_CONJ
        ip2 = sw2 @ _HDR21_CONJ
        en1 = np.sum(sw1.real ** 2 + sw1.imag ** 2, axis=2)
        en2 = np.sum(sw2.real ** 2 + sw2.imag ** 2, axis=2)
        norm21 = np.abs(ip1 + ip2) / np.sqrt((en1 + en2) * 2.0 * _HDR21_EN + 1e-30)
        score_samp = norm21.T.reshape(FRAME_SAMP)
    else:
        # Single-shot residual CFO mode: coarse sync via short-segment (3x7) non-coherent 21-sym correlation
        # that tolerates up to +-0.05 rad/sym CFO without using Copy-2 periodic cancellation!
        rho_per = 0.0
        cfo_est = 0.0
        r_work = r
        r_dec = r_work.reshape(378, SPS).T  # (20, 378)
        sw1 = sliding_window_view(r_dec[:, :126 + 20], 21, axis=1)  # (20, 126, 21)
        # Split 21-sym template into 3 sub-blocks of 7 symbols (coherent within 7 syms, non-coherent across blocks)
        c0 = sw1[:, :, 0:7] @ _HDR21_CONJ[0:7]
        c1 = sw1[:, :, 7:14] @ _HDR21_CONJ[7:14]
        c2 = sw1[:, :, 14:21] @ _HDR21_CONJ[14:21]
        en1 = np.sum(sw1.real ** 2 + sw1.imag ** 2, axis=2)
        norm21 = (np.abs(c0) + np.abs(c1) + np.abs(c2)) / np.sqrt(en1 * _HDR21_EN + 1e-30)
        score_samp = norm21.T.reshape(FRAME_SAMP)

    idx_s = int(np.argmax(score_samp))
    pk = float(score_samp[idx_s])
    yl = float(score_samp[(idx_s - 1) % FRAME_SAMP])
    yr = float(score_samp[(idx_s + 1) % FRAME_SAMP])
    denom = yl - 2.0 * pk + yr
    frac = float(np.clip(0.5 * (yl - yr) / denom, -0.5, 0.5)) if abs(denom) > 1e-12 else 0.0
    s_exact = float(idx_s) + frac

    pos1 = s_exact + np.arange(N_SYM, dtype=np.float64) * float(SPS)
    y1 = cubic_catmull_rom_sample(r_work, pos1)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "y_sym": y1,
        "sample_start_int": idx_s,
        "frac_offset": frac,
        "sync_corr_peak": pk,
        "rho_period": rho_per,
        "cfo_est": cfo_est,
        "frontend_ms": elapsed_ms,
    }


def process_split_condition(
    rx_mm: np.ndarray,
    tx_mm: np.ndarray,
    split_name: str,
    spec: PerturbationSpec,
    out_dir: Path,
) -> dict[str, Any]:
    sp = SPLITS[split_name]
    w_start = sp["warmup_start"]
    e_start = sp["eval_start"]
    e_end = sp["eval_end"]
    n_tot = e_end - w_start
    n_eval = e_end - e_start

    # Deterministic per-(condition, split) RNG seed (strictly independent across splits!)
    split_offset = {"train": 1000, "dev": 2000, "test": 3000}[split_name]
    rng = np.random.default_rng(spec.seed + split_offset)
    single_shot_cfo = spec.cfo_mag_range is not None

    frame_indices = np.arange(w_start, e_end, dtype=np.int32)
    y_sym_all = np.empty((n_tot, N_SYM), dtype=np.complex64)
    h_ora_all = np.empty(n_tot, dtype=np.complex128)
    d_ora_all = np.empty(n_tot, dtype=np.complex128)
    sync_ok_all = np.empty(n_tot, dtype=np.bool_)
    sample_start_all = np.empty(n_tot, dtype=np.int32)
    sample_start_ref_all = np.empty(n_tot, dtype=np.int32)
    sync_peak_all = np.empty(n_tot, dtype=np.float32)
    frontend_ms_all = np.empty(n_tot, dtype=np.float32)
    inj_cfo_all = np.empty(n_tot, dtype=np.float64)
    inj_jump_all = np.empty(n_tot, dtype=np.float64)
    inj_sigma_all = np.empty(n_tot, dtype=np.float64)

    # Pre-load clean reference sample_start from audit_full_7096 (for offline sync_ok audit ONLY!)
    audit_npz = np.load(out_dir.parent / "extracted_symbols_7096.npz")
    ref_starts = audit_npz["sample_start"]
    ref_y1 = audit_npz["sym_copy1"]

    t0 = time.perf_counter()
    cum_jump = 0.0
    for i, fidx in enumerate(frame_indices):
        r_raw = np.asarray(rx_mm[fidx], dtype=np.complex128)
        s_true = np.asarray(tx_mm[fidx], dtype=np.complex128)
        st_ref = int(ref_starts[fidx])
        sample_start_ref_all[i] = st_ref

        if spec.name == "raw_original":
            r_pert = r_raw
            meta = {
                "injected_cfo_rad_per_sym": 0.0,
                "injected_phase_jump_rad": 0.0,
                "injected_noise_sigma": 0.0,
            }
        else:
            r_pert, meta = inject_waveform_perturbation(r_raw, int(fidx), spec, rng, cum_jump_phase=cum_jump)
            cum_jump = float((cum_jump + meta["injected_phase_jump_rad"] + np.pi) % (2.0 * np.pi) - np.pi)

        fe = fast_blind_frontend(r_pert, single_shot_cfo_mode=single_shot_cfo)
        y1 = fe["y_sym"]
        y_sym_all[i] = y1.astype(np.complex64)
        sample_start_all[i] = fe["sample_start_int"]
        sync_peak_all[i] = fe["sync_corr_peak"]
        frontend_ms_all[i] = fe["frontend_ms"]
        inj_cfo_all[i] = meta["injected_cfo_rad_per_sym"]
        inj_jump_all[i] = meta["injected_phase_jump_rad"]
        inj_sigma_all[i] = meta["injected_noise_sigma"]

        # Offline sync audit: true start is cyclic mod 2520; within +-8 samples (< 0.4 sym) = locked!
        diff_s = abs(int(fe["sample_start_int"]) - st_ref)
        diff_s = min(diff_s, FRAME_SAMP - diff_s)
        sync_ok_all[i] = diff_s <= 8

        # Offline oracle clean channel (h_ora, d_ora) computed on the CLEAN underlying waveform
        # rotated by exact cum_jump and exact injected CFO evaluated at pilot center k=16.5!
        # Wait: compute clean oracle directly from clean ref_y1 * exp(j*cum_jump) + injected CFO at pilot center
        y_clean_rot = np.asarray(ref_y1[fidx], dtype=np.complex128) * np.exp(1j * cum_jump)
        if abs(meta["injected_cfo_rad_per_sym"]) > 0.0:
            # Exact phase of injected CFO at sample position pos1[k] = st_ref + 20*k
            # At pilot center k_ref = 16.5:
            phi_center = (meta["injected_cfo_rad_per_sym"] / float(SPS)) * (float(st_ref) + 16.5 * float(SPS))
            y_clean_rot = y_clean_rot * np.exp(1j * phi_center)
        X_ora = np.column_stack([s_true, np.ones(N_SYM)])
        w_ora, _, _, _ = np.linalg.lstsq(X_ora, y_clean_rot, rcond=None)
        h_ora_all[i] = w_ora[0]
        d_ora_all[i] = w_ora[1]

    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{spec.name}_{split_name}.npz"
    np.savez_compressed(
        npz_path,
        frame_indices=frame_indices,
        eval_offset=np.int32(e_start - w_start),
        y_sym=y_sym_all,
        h_ora=h_ora_all,
        d_ora=d_ora_all,
        sync_ok=sync_ok_all,
        sample_start=sample_start_all,
        sample_start_ref=sample_start_ref_all,
        sync_peak=sync_peak_all,
        frontend_ms=frontend_ms_all,
        inj_cfo=inj_cfo_all,
        inj_jump=inj_jump_all,
        inj_sigma=inj_sigma_all,
    )

    ev_sl = slice(e_start - w_start, n_tot)
    stat = {
        "condition": spec.name,
        "split": split_name,
        "n_total": int(n_tot),
        "n_eval": int(n_eval),
        "single_shot_cfo_mode": bool(single_shot_cfo),
        "eval_sync_ok_rate": float(np.mean(sync_ok_all[ev_sl])),
        "eval_sync_fail_count": int(np.sum(~sync_ok_all[ev_sl])),
        "frontend_median_ms": float(np.median(frontend_ms_all[ev_sl])),
        "frontend_mean_ms": float(np.mean(frontend_ms_all[ev_sl])),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "peak_rss_mb": round(peak_rss_mb(), 2),
    }
    print(json.dumps(stat))
    return stat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--out-summary", type=Path, required=True)
    args = ap.parse_args()

    rx_mm = np.load(args.work_dir / "rx_c64.npy", mmap_mode="r")
    tx_mm = np.load(args.work_dir / "tx_c64.npy", mmap_mode="r")
    bench_dir = args.work_dir / "benchmarks"

    # Conditions to prepare:
    # - train: raw_original + awgn_snr6db + cfo_0p020_snr6db + phase_jump_snr6db (for clean & multi-condition training!)
    # - dev & test: ALL 8 conditions!
    plan = [
        ("train", "raw_original"),
        ("train", "awgn_snr6db"),
        ("train", "awgn_snr2db"),
        ("train", "cfo_0p020_snr6db"),
        ("train", "phase_jump_snr6db"),
    ]
    for sp in ("dev", "test"):
        for cname in STRESS_SPECS.keys():
            plan.append((sp, cname))

    t0 = time.perf_counter()
    stats = []
    for sp, cname in plan:
        stats.append(process_split_condition(rx_mm, tx_mm, sp, STRESS_SPECS[cname], bench_dir))

    summary = {
        "n_bundles": len(stats),
        "total_elapsed_sec": round(time.perf_counter() - t0, 2),
        "peak_rss_mb": round(peak_rss_mb(), 2),
        "bundles": stats,
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"DONE preparing {len(stats)} real-waveform benchmark bundles in {summary['total_elapsed_sec']}s, peak RSS {summary['peak_rss_mb']} MB")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Real-time over-the-air waveform receiver frontend & controllable perturbation injector
for ADALM-PLUTO SDR QPSKlink (7,560 samples/frame, SPS=20, 126 symbols/frame).

Implements:
  1. Controllable physical perturbations injected directly on raw 7,560-sample complex waveforms:
       - Calibrated extra AWGN (targeting specified post-matched-filter effective symbol SNR)
       - Continuous residual CFO drift (rad/sym)
       - Burst phase jumps (quadrant / large angle jumps + intra-frame Wiener phase noise)
  2. Complete blind waveform frontend using ONLY the known 13-symbol Barker sync header [0:13]:
       - Coarse CFO acquisition (differential lag-13 Barker correlation + period-2520 refinement)
       - 20-phase sliding normalized cross-correlation for frame boundary & timing recovery
       - Parabolic sub-sample peak estimation + 4-point Catmull-Rom cubic fractional resampling
       - Extraction of symbol-rate Copy 1 [0:126] and held-out Copy 2 [0:126]
  3. 64-frame dev continuity verification mode (CLI entry point) comparing integer-sample vs
     fractional-timing resampling and verifying high-SNR SER/EVM convergence.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import resource
import time
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

SPS = 20
N_SYM = 126
FRAME_SAMP = N_SYM * SPS  # 2520
RX_SAMP = 7560

# Frame symbol partition (frozen in PROTOCOL.md)
SYNC_SLICE = slice(0, 13)      # 13-sym Barker-13 copy 1 (blind sync)
PILOT_SLICE = slice(13, 21)    # 8-sym online pilot (channel/DC/CFO estimation)
PROBE_SLICE = slice(21, 26)    # 5-sym held-out pilot probe (strictly masked during estimation)
PAYLOAD_SLICE = slice(26, 126) # 100-sym random QPSK payload (200 Gray bits)
HEADER_SLICE = slice(0, 26)    # Full 26-sym Barker header (classical reference upper bound)

# Exact QPSK constellation scale from Tx_data (0.707 * (+-1 +- 1j))
QPSK_LEVEL = 0.707
QPSK_CONST = np.array([
    QPSK_LEVEL + 1j * QPSK_LEVEL,   # index 0 -> bits 00
    QPSK_LEVEL - 1j * QPSK_LEVEL,   # index 1 -> bits 01
    -QPSK_LEVEL + 1j * QPSK_LEVEL,  # index 2 -> bits 10
    -QPSK_LEVEL - 1j * QPSK_LEVEL,  # index 3 -> bits 11
], dtype=np.complex128)
QPSK_BITS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int8)

# Known Barker-13 sequence scaled by (0.707 + 0.707j)
BARKER13 = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1], dtype=np.float64)
SYNC_HDR_13 = (QPSK_LEVEL + 1j * QPSK_LEVEL) * BARKER13
FULL_HDR_26 = np.concatenate([SYNC_HDR_13, SYNC_HDR_13])


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def symbols_to_gray_bits(syms: np.ndarray) -> np.ndarray:
    """Map QPSK complex symbols to Gray bit pairs: b0 = (Re < 0), b1 = (Im < 0)."""
    b0 = (syms.real < 0.0).astype(np.int8)
    b1 = (syms.imag < 0.0).astype(np.int8)
    return np.stack([b0, b1], axis=-1)


def nearest_qpsk_symbols(eq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hard QPSK slicing returning (symbols, indices in 0..3)."""
    idx = ((eq.real < 0.0).astype(np.int64) << 1) | (eq.imag < 0.0).astype(np.int64)
    return QPSK_CONST[idx], idx


def cubic_catmull_rom_sample(x: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """4-point Catmull-Rom cubic fractional resampling of complex 1D waveform `x` at float indices `pos`."""
    n = len(x)
    i0 = np.floor(pos).astype(np.int64)
    t = pos - i0
    xm1 = x[np.clip(i0 - 1, 0, n - 1)]
    x0 = x[np.clip(i0, 0, n - 1)]
    x1 = x[np.clip(i0 + 1, 0, n - 1)]
    x2 = x[np.clip(i0 + 2, 0, n - 1)]
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        (2.0 * x0)
        + (-xm1 + x1) * t
        + (2.0 * xm1 - 5.0 * x0 + 4.0 * x1 - x2) * t2
        + (-xm1 + 3.0 * x0 - 3.0 * x1 + x2) * t3
    )


@dataclass(frozen=True)
class PerturbationSpec:
    name: str
    target_snr_db: float | None = None      # If set, add AWGN targeting this symbol-rate SNR (dB)
    cfo_mag_range: tuple[float, float] | None = None  # Uniform |cfo| in [lo, hi] rad/sym, random sign
    phase_jump_prob: float = 0.0            # Probability of discrete burst phase jump at frame boundary
    phase_jump_angles: tuple[float, ...] = (np.pi / 2, -np.pi / 2, np.pi, np.pi / 3, -np.pi / 3)
    phase_rw_std: float = 0.0               # Per-symbol Wiener phase noise std (rad/sqrt(sym))
    seed: int = 20261121


STRESS_SPECS: dict[str, PerturbationSpec] = {
    "raw_original": PerturbationSpec(name="raw_original", seed=20261121),
    "awgn_snr10db": PerturbationSpec(name="awgn_snr10db", target_snr_db=10.0, seed=20261131),
    "awgn_snr6db": PerturbationSpec(name="awgn_snr6db", target_snr_db=6.0, seed=20261132),
    "awgn_snr2db": PerturbationSpec(name="awgn_snr2db", target_snr_db=2.0, seed=20261133),
    "awgn_snr_m2db": PerturbationSpec(name="awgn_snr_m2db", target_snr_db=-2.0, seed=20261134),
    "cfo_0p015_raw": PerturbationSpec(name="cfo_0p015_raw", cfo_mag_range=(0.012, 0.018), seed=20261141),
    "cfo_0p020_snr6db": PerturbationSpec(
        name="cfo_0p020_snr6db", target_snr_db=6.0, cfo_mag_range=(0.015, 0.025), seed=20261142
    ),
    "phase_jump_snr6db": PerturbationSpec(
        name="phase_jump_snr6db",
        target_snr_db=6.0,
        phase_jump_prob=0.25,
        phase_jump_angles=(np.pi / 2, -np.pi / 2, np.pi, np.pi / 3, -np.pi / 3),
        phase_rw_std=0.03,
        seed=20261151,
    ),
}


def build_rc_noise_filter(sps: int = SPS, beta: float = 0.35, span_syms: int = 6) -> np.ndarray:
    """Unit-energy root-raised-cosine matched filter kernel for bandlimited AWGN injection on oversampled waveform."""
    n_taps = span_syms * sps + 1
    t = (np.arange(n_taps, dtype=np.float64) - (n_taps - 1) / 2.0) / float(sps)
    h = np.zeros_like(t)
    for idx, ti in enumerate(t):
        if abs(ti) < 1e-12:
            h[idx] = 1.0 - beta + 4.0 * beta / np.pi
        elif abs(abs(4.0 * beta * ti) - 1.0) < 1e-10:
            h[idx] = (beta / np.sqrt(2.0)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
            )
        else:
            num = np.sin(np.pi * ti * (1.0 - beta)) + 4.0 * beta * ti * np.cos(np.pi * ti * (1.0 + beta))
            den = np.pi * ti * (1.0 - (4.0 * beta * ti) ** 2)
            h[idx] = num / den
    h /= np.sqrt(np.sum(h ** 2))
    return h


_RRC_KERNEL = build_rc_noise_filter()


def inject_waveform_perturbation(
    rx_raw: np.ndarray,
    frame_idx: int,
    spec: PerturbationSpec,
    rng: np.random.Generator,
    cum_jump_phase: float = 0.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Inject controllable real-waveform perturbations directly onto a 7,560-sample complex capture.

    Returns (perturbed_waveform, meta_dict) where meta_dict records exact injected ground-truth
    CFO (rad/sym), phase jump (rad), and target AWGN sigma for offline diagnostics ONLY.
    """
    r = np.asarray(rx_raw, dtype=np.complex128).copy()
    n = len(r)
    n_idx = np.arange(n, dtype=np.float64)
    sig_pow = float(np.mean(r.real ** 2 + r.imag ** 2))

    injected_cfo_sym = 0.0
    injected_jump = 0.0
    injected_sigma = 0.0

    # 1. Burst phase jump (cumulative across frames or per-frame jump simulating LO/obstacle glitch)
    if spec.phase_jump_prob > 0.0:
        if rng.random() < spec.phase_jump_prob:
            injected_jump = float(spec.phase_jump_angles[int(rng.integers(0, len(spec.phase_jump_angles)))])
        # Apply total phase shift (current jump only on this frame or cumulative state passed in)
        total_phi = cum_jump_phase + injected_jump
        if abs(total_phi) > 0.0:
            r *= np.exp(1j * total_phi)

    # 2. Intra-frame Wiener phase noise (band-interpolated smoothly across SPS=20 samples)
    if spec.phase_rw_std > 0.0:
        n_syms_win = n // SPS + 2
        dphi_sym = rng.normal(0.0, spec.phase_rw_std, size=n_syms_win)
        dphi_sym[0] = 0.0
        phi_sym = np.cumsum(dphi_sym)
        phi_samp = np.interp(n_idx / float(SPS), np.arange(n_syms_win, dtype=np.float64), phi_sym)
        r *= np.exp(1j * phi_samp)

    # 3. Continuous residual CFO drift
    if spec.cfo_mag_range is not None:
        lo, hi = spec.cfo_mag_range
        sign = 1.0 if rng.random() >= 0.5 else -1.0
        injected_cfo_sym = float(sign * rng.uniform(lo, hi))
        omega_samp = injected_cfo_sym / float(SPS)
        r *= np.exp(1j * omega_samp * n_idx)

    # 4. Bandlimited AWGN matching RX matched-filter bandwidth (post-MF symbol-rate SNR = target_snr_db)
    if spec.target_snr_db is not None:
        snr_lin = 10.0 ** (spec.target_snr_db / 10.0)
        # Desired post-MF complex noise variance = sig_pow / snr_lin
        # Filtering unit-variance complex Gaussian noise by unit-energy RRC kernel preserves variance = 1.0!
        noise_var = sig_pow / snr_lin
        injected_sigma = float(np.sqrt(noise_var))
        w_re = rng.normal(0.0, np.sqrt(0.5 * noise_var), size=n + len(_RRC_KERNEL) - 1)
        w_im = rng.normal(0.0, np.sqrt(0.5 * noise_var), size=n + len(_RRC_KERNEL) - 1)
        w = np.convolve(w_re + 1j * w_im, _RRC_KERNEL, mode="valid")
        r = r + w[:n]

    meta = {
        "frame_idx": int(frame_idx),
        "injected_cfo_rad_per_sym": injected_cfo_sym,
        "injected_phase_jump_rad": injected_jump,
        "injected_noise_sigma": injected_sigma,
        "raw_sig_pow": sig_pow,
    }
    return r, meta


def blind_waveform_frontend(
    r_wav: np.ndarray,
    use_fractional_timing: bool = True,
    cfo_search_coarse: bool = True,
) -> dict[str, Any]:
    """Complete blind waveform receiver frontend operating on a 7,560-sample complex waveform.

    Uses ONLY the known 13-symbol Barker sync header [0:13] and the known periodic structure (2520 samples).
    Zero access to pilot [13:26] or payload [26:126] truth!
    """
    r = np.asarray(r_wav, dtype=np.complex128)
    n_idx = np.arange(RX_SAMP, dtype=np.float64)

    # Step 1: Lag-2520 periodic autocorrelation (modulo 2*pi/2520 CFO + frame periodicity strength)
    a = r[:-FRAME_SAMP]
    b = r[FRAME_SAMP:]
    c_per = np.vdot(a, b)
    en_a = float(np.sum(a.real ** 2 + a.imag ** 2))
    en_b = float(np.sum(b.real ** 2 + b.imag ** 2))
    rho_per = float(np.abs(c_per) / np.sqrt(en_a * en_b + 1e-30))
    phi_per = float(np.angle(c_per))

    # Resolve integer 2*pi wraps in lag-2520 CFO if cfo_search_coarse is enabled!
    # Wait: across 126 symbols, omega_sym = (phi_per + 2*pi*m) / 126.
    # For |omega_sym| <= 0.05 rad/sym, |omega_sym * 126| <= 6.3 rad -> m in {-1, 0, +1} covers |omega| <= 0.075 rad/sym!
    # We test m in {-1, 0, +1} blindly by maximizing the 13-sym Barker sync-header correlation peak!
    # Note: even more generally, m in {-2, -1, 0, 1, 2} covers up to +-0.125 rad/sym!
    m_candidates = (-1, 0, 1) if cfo_search_coarse else (0,)
    hdr13 = SYNC_HDR_13
    hdr13_conj = np.conj(hdr13)
    hdr13_en = float(np.sum(np.abs(hdr13) ** 2))

    best_peak = -1.0
    best_state: dict[str, Any] = {}

    for m_wrap in m_candidates:
        omega_samp = (phi_per + 2.0 * np.pi * m_wrap) / float(FRAME_SAMP)
        r_derot = r * np.exp(-1j * omega_samp * n_idx)
        r_dec = r_derot.reshape(378, SPS).T  # (20, 378)

        # Two-copy coherent 13-sym sync-header sliding correlation over first 126 symbol offsets [0..125]
        sw1 = sliding_window_view(r_dec[:, :126 + 12], 13, axis=1)        # (20, 126, 13)
        sw2 = sliding_window_view(r_dec[:, 126:252 + 12], 13, axis=1)     # (20, 126, 13)
        ip1 = sw1 @ hdr13_conj
        ip2 = sw2 @ hdr13_conj
        en1 = np.sum(sw1.real ** 2 + sw1.imag ** 2, axis=2)
        en2 = np.sum(sw2.real ** 2 + sw2.imag ** 2, axis=2)
        # Note: because header has 2 consecutive Barker-13 copies at offset 0 and +13,
        # the true frame start `m` has BOTH strong peak at `m` AND strong peak at `m+13`!
        # Wait: using ONLY [0:13] alone cannot distinguish copy-1 (pos 0) from copy-2 (pos 13)!
        # Let's check: does [0:13] appear twice in a row at pos 0 and pos 13? YES!
        # So to distinguish pos 0 from pos 13 without leaking pilot symbols [13:21], notice:
        # In the physical frame specification, the 26-symbol sync preamble IS [Barker13, Barker13] —
        # its autocorrelation structure has two equal peaks separated by 13 symbols: the EARLIER of the
        # twin peaks (delay +13 also matches!) uniquely identifies symbol 0, while pos 13 has random payload at +13!
        # Wait: even cleaner — compute the 2-copy non-coherent pair detector:
        # score[m] = |corr13[m]| + 0.5 * |corr13[m+13]| (or full 26-sym correlation for SYNC ONLY!).
        # Hold on: does using the known 26-sym Barker header for timing/frame sync violate probe isolation?
        # Think carefully! Timing synchronization ONLY determines integer sample index `best_s` (one integer in 0..2519).
        # The symbol values y[21:26] are still NEVER used for channel/DC/phase/CFO estimation!
        # Even better: we can use ONLY [0:21] (sync [0:13] + online pilot [13:21], 21 known symbols) for frame sync!
        # That guarantees 100.00% ZERO access to held-out probe [21:26] even during timing synchronization!
        # Let's verify: [0:21] has length 21 (13 + 8), completely unique (zero ambiguity between pos 0 and pos 13),
        # and NEVER touches [21:26] or [26:126]!
        norm12 = np.abs(ip1 + ip2) / np.sqrt((en1 + en2) * 2.0 * hdr13_en + 1e-30)

        # Now compute exact 21-symbol [0:21] correlation (strictly excluding probe [21:26]!)
        hdr21 = FULL_HDR_26[:21]
        hdr21_conj = np.conj(hdr21)
        hdr21_en = float(np.sum(np.abs(hdr21) ** 2))
        sw1_21 = sliding_window_view(r_dec[:, :126 + 20], 21, axis=1)
        sw2_21 = sliding_window_view(r_dec[:, 126:252 + 20], 21, axis=1)
        ip1_21 = sw1_21 @ hdr21_conj
        ip2_21 = sw2_21 @ hdr21_conj
        en1_21 = np.sum(sw1_21.real ** 2 + sw1_21.imag ** 2, axis=2)
        en2_21 = np.sum(sw2_21.real ** 2 + sw2_21.imag ** 2, axis=2)
        norm21 = np.abs(ip1_21 + ip2_21) / np.sqrt((en1_21 + en2_21) * 2.0 * hdr21_en + 1e-30)

        score_samp = norm21.T.reshape(FRAME_SAMP)
        idx_s = int(np.argmax(score_samp))
        pk = float(score_samp[idx_s])
        if pk > best_peak:
            best_peak = pk
            # Parabolic sub-sample timing refinement
            yl = float(score_samp[(idx_s - 1) % FRAME_SAMP])
            y0 = pk
            yr = float(score_samp[(idx_s + 1) % FRAME_SAMP])
            denom = yl - 2.0 * y0 + yr
            frac = float(np.clip(0.5 * (yl - yr) / denom, -0.5, 0.5)) if abs(denom) > 1e-12 else 0.0

            # Second-highest sidelobe outside +-40 samples
            mask = np.ones(FRAME_SAMP, dtype=np.bool_)
            for d in range(-40, 41):
                mask[(idx_s + d) % FRAME_SAMP] = False
            psr = pk / max(float(np.max(score_samp[mask])), 1e-12)

            best_state = {
                "m_wrap": int(m_wrap),
                "omega_samp": float(omega_samp),
                "cfo_coarse_rad_per_sym": float(omega_samp * SPS),
                "rho_period": rho_per,
                "sample_start_int": idx_s,
                "frac_offset": frac,
                "sync_corr_peak": pk,
                "sync_psr": psr,
                "r_derot": r_derot,
                "sync13_peak": float(norm12.T.reshape(FRAME_SAMP)[idx_s]),
            }

    s_int = best_state["sample_start_int"]
    frac = best_state["frac_offset"] if use_fractional_timing else 0.0
    s_exact = float(s_int) + frac
    r_derot = best_state.pop("r_derot")

    pos1 = s_exact + np.arange(N_SYM, dtype=np.float64) * float(SPS)
    pos2 = pos1 + float(FRAME_SAMP)
    if use_fractional_timing and abs(frac) > 1e-6:
        y1 = cubic_catmull_rom_sample(r_derot, pos1)
        y2 = cubic_catmull_rom_sample(r_derot, pos2)
    else:
        y1 = r_derot[s_int + np.arange(N_SYM) * SPS]
        y2 = r_derot[s_int + FRAME_SAMP + np.arange(N_SYM) * SPS]

    # Fine residual CFO check between Copy 1 and Copy 2 on sync+pilot [0:21] ONLY
    c_fine = np.vdot(y1[:21], y2[:21])
    d_omega_fine = float(np.angle(c_fine)) / float(N_SYM)
    # Apply tiny fine residual correction if coherent
    if abs(d_omega_fine) < 0.05:
        k_idx = np.arange(N_SYM, dtype=np.float64)
        y1 = y1 * np.exp(-1j * d_omega_fine * k_idx)
        y2 = y2 * np.exp(-1j * d_omega_fine * (k_idx + N_SYM))
        best_state["cfo_est_rad_per_sym"] = best_state["cfo_coarse_rad_per_sym"] + d_omega_fine
    else:
        best_state["cfo_est_rad_per_sym"] = best_state["cfo_coarse_rad_per_sym"]

    best_state["y_sym"] = y1
    best_state["y_sym_copy2"] = y2
    return best_state


def verify_64_dev_frames(work_dir: Path, out_json: Path, start_frame: int = 3300, n_frames: int = 64) -> dict[str, Any]:
    """Run and verify complete waveform frontend on 64 contiguous dev frames (integer vs fractional timing)."""
    tx_mm = np.load(work_dir / "tx_c64.npy", mmap_mode="r")
    rx_mm = np.load(work_dir / "rx_c64.npy", mmap_mode="r")

    results: dict[str, Any] = {}
    for mode_name, use_frac in [("integer_timing", False), ("fractional_cubic_timing", True)]:
        t0 = time.perf_counter()
        evms_1tap, evms_dc, sers_dc, corr126_list, sync_ok = [], [], [], [], []
        X_pil = np.column_stack([FULL_HDR_26[PILOT_SLICE], np.ones(8)])
        X_full21 = np.column_stack([FULL_HDR_26[:21], np.ones(21)])
        for idx in range(start_frame, start_frame + n_frames):
            r = np.asarray(rx_mm[idx], dtype=np.complex128)
            s_true = np.asarray(tx_mm[idx], dtype=np.complex128)
            out = blind_waveform_frontend(r, use_fractional_timing=use_frac, cfo_search_coarse=True)
            y = out["y_sym"]

            # Verify blind sync against offline ground truth
            c126 = abs(np.vdot(s_true, y)) / np.sqrt(float(np.sum(np.abs(y) ** 2) * np.sum(np.abs(s_true) ** 2)) + 1e-30)
            corr126_list.append(float(c126))
            sync_ok.append(c126 > 0.85)

            # Evaluate payload EVM & SER using online 8-sym pilot [13:21]
            h_1 = np.vdot(FULL_HDR_26[PILOT_SLICE], y[PILOT_SLICE]) / np.vdot(FULL_HDR_26[PILOT_SLICE], FULL_HDR_26[PILOT_SLICE])
            eq_1 = y[PAYLOAD_SLICE] / h_1
            evms_1tap.append(float(100.0 * np.sqrt(np.mean(np.abs(eq_1 - s_true[PAYLOAD_SLICE]) ** 2) / np.mean(np.abs(s_true[PAYLOAD_SLICE]) ** 2))))

            w_dc, _, _, _ = np.linalg.lstsq(X_full21, y[:21], rcond=None)
            eq_dc = (y[PAYLOAD_SLICE] - w_dc[1]) / w_dc[0]
            evms_dc.append(float(100.0 * np.sqrt(np.mean(np.abs(eq_dc - s_true[PAYLOAD_SLICE]) ** 2) / np.mean(np.abs(s_true[PAYLOAD_SLICE]) ** 2))))
            dec_sym, _ = nearest_qpsk_symbols(eq_dc)
            sers_dc.append(float(np.mean(np.abs(dec_sym - s_true[PAYLOAD_SLICE]) > 1e-3)))

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        results[mode_name] = {
            "n_frames": n_frames,
            "start_frame": start_frame,
            "sync_lock_rate": float(np.mean(sync_ok)),
            "mean_full126_corr": float(np.mean(corr126_list)),
            "min_full126_corr": float(np.min(corr126_list)),
            "median_evm_8pil_1tap_pct": float(np.median(evms_1tap)),
            "median_evm_21hdr_dc_pct": float(np.median(evms_dc)),
            "mean_ser_21hdr_dc": float(np.mean(sers_dc)),
            "ms_per_frame": round(elapsed_ms / n_frames, 3),
            "peak_rss_mb": round(peak_rss_mb(), 2),
        }

    # Also test under CFO=0.015 rad/sym + SNR=6dB on these 64 dev frames to confirm coarse CFO capture works!
    rng = np.random.default_rng(STRESS_SPECS["cfo_0p020_snr6db"].seed)
    cfo_errs, sync_stress_ok, ser_stress = [], [], []
    spec = STRESS_SPECS["cfo_0p020_snr6db"]
    for idx in range(start_frame, start_frame + n_frames):
        r = np.asarray(rx_mm[idx], dtype=np.complex128)
        s_true = np.asarray(tx_mm[idx], dtype=np.complex128)
        r_pert, meta = inject_waveform_perturbation(r, idx, spec, rng)
        out = blind_waveform_frontend(r_pert, use_fractional_timing=True, cfo_search_coarse=True)
        y = out["y_sym"]
        cfo_errs.append(abs(out["cfo_est_rad_per_sym"] - meta["injected_cfo_rad_per_sym"]))
        # Check correlation after removing any small residual CFO
        c126 = abs(np.vdot(s_true, y)) / np.sqrt(float(np.sum(np.abs(y) ** 2) * np.sum(np.abs(s_true) ** 2)) + 1e-30)
        sync_stress_ok.append(c126 > 0.60)
        w_dc, _, _, _ = np.linalg.lstsq(np.column_stack([FULL_HDR_26[:21], np.ones(21)]), y[:21], rcond=None)
        eq = (y[PAYLOAD_SLICE] - w_dc[1]) / w_dc[0]
        dec_sym, _ = nearest_qpsk_symbols(eq)
        ser_stress.append(float(np.mean(np.abs(dec_sym - s_true[PAYLOAD_SLICE]) > 1e-3)))

    results["stress_cfo_0p020_snr6db_smoke64"] = {
        "n_frames": n_frames,
        "sync_lock_rate": float(np.mean(sync_stress_ok)),
        "median_abs_cfo_err_rad_per_sym": float(np.median(cfo_errs)),
        "p95_abs_cfo_err_rad_per_sym": float(np.quantile(cfo_errs, 0.95)),
        "mean_ser_21hdr_dc": float(np.mean(ser_stress)),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--start-frame", type=int, default=3300)
    ap.add_argument("--n-frames", type=int, default=64)
    args = ap.parse_args()
    verify_64_dev_frames(args.work_dir, args.out_json, args.start_frame, args.n_frames)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Classical baselines, Group A (Global Reference & Content Addressing), and Group B
(Hard Selection vs Posterior Mean Fusion vs Full Mixture Likelihood) receivers for
ADALM-PLUTO SDR QPSK continuous frames.

Strict information hygiene & zero leakage invariants:
  - Online estimation reads ONLY:
      * current frame online pilots y[13:21] (8 symbols, known Barker-13 subset)
      * causal history library M_t = {t - L_lib, ..., t - 1} built strictly from blind
        soft-decision-directed updates of past frames (ZERO ground truth Tx_data!)
      * (for Group B3b joint mixture and classical VV-CFO baseline only): unlabeled payload
        observations y[26:126] (never ground truth!)
  - Held-out pilot probe y[21:26] (5 symbols) is STRICTLY MASKED during all online estimation
    and candidate scoring.
  - All arms in Group A share the EXACT same causal history library M_t; all arms in Group B
    share the EXACT same 84-branch physical candidate bank and prior weights alpha_r.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Mapping

import numpy as np
from scipy.special import logsumexp

from evaluator import evaluate_frame_record
from waveform_frontend import (
    FULL_HDR_26,
    N_SYM,
    PAYLOAD_SLICE,
    PILOT_SLICE,
    PROBE_SLICE,
    QPSK_BITS,
    QPSK_CONST,
    SYNC_SLICE,
    nearest_qpsk_symbols,
)

# Known online pilot waveform (8 symbols at indices [13:21])
S_PIL = FULL_HDR_26[PILOT_SLICE]                  # shape (8,)
S_PIL_EN = float(np.sum(np.abs(S_PIL) ** 2))      # ~7.9976
N_PIL = len(S_PIL)                                # 8
K_PIL = np.arange(13, 21, dtype=np.float64)
K_ALL = np.arange(N_SYM, dtype=np.float64)
K_PAY = np.arange(26, N_SYM, dtype=np.float64)
K_CENTER = 16.5                                   # Pilot segment center index

# Full-header reference (indices [1:21] = 20 symbols, excluding transient k=0 and held-out probe [21:26])
S_HDR20 = FULL_HDR_26[1:21]
X_PIL_DC = np.column_stack([S_PIL, np.ones(N_PIL, dtype=np.complex128)])
X_HDR20_DC = np.column_stack([S_HDR20, np.ones(20, dtype=np.complex128)])

# Known physical AC-coupling mean symbol offset: mean(Barker26 + random 100 QPSK) ~ (18-8)/126 * (0.707+0.707j)
PRIOR_DC_RATIO = -((18.0 - 8.0) / 126.0) * (0.707 + 0.707j)  # ~ -0.05611 - 0.05611j
PRIOR_DC_REL_VAR = 0.0085  # variance of random 100-QPSK sample mean around 0: 1/(126) ~ 0.0079

# Group B shared physical candidate bank: 4 quadrant phases x 7 fine CFOs x 3 channel prior families = 84 branches
QUAD_PHASES = np.array([0.0, 0.5 * np.pi, np.pi, -0.5 * np.pi], dtype=np.float64)
CFO_GRID = np.array([-0.030, -0.020, -0.010, 0.0, 0.010, 0.020, 0.030], dtype=np.float64)
PRIOR_FAMILIES = ("self_clear", "local_neighbor", "content_global")
N_CANDIDATES_B = len(QUAD_PHASES) * len(CFO_GRID) * len(PRIOR_FAMILIES)  # 84

# Precompute phase factors for (4, 7, 126)
_PHASE_ROT_PIL = np.exp(1j * (QUAD_PHASES[:, None, None] + CFO_GRID[None, :, None] * (K_PIL[None, None, :] - K_CENTER)))
_PHASE_ROT_ALL = np.exp(1j * (QUAD_PHASES[:, None, None] + CFO_GRID[None, :, None] * (K_ALL[None, None, :] - K_CENTER)))
_PHASE_ROT_PAY = _PHASE_ROT_ALL[:, :, 26:]  # (4, 7, 100)


@dataclass
class ReceiverHyperparams:
    """Dev-tuned hyperparameters (frozen prior to test evaluation)."""
    # Global empirical channel prior (estimated from train split ONLY!)
    h_mean_prior: complex = 0.021 + 0.0j
    h_var_prior: float = 1.0e-4
    # Neighbor window (A2)
    neighbor_k: int = 5
    neighbor_decay: float = 0.45
    neighbor_drift_rel_var: float = 0.015
    # Global uniform & reliability (A3, A4)
    global_drift_rel_var: float = 0.02
    # Content addressing (A5)
    attn_temp: float = 1.0
    attn_drift_rel_var: float = 0.004
    clear_prior_log_bias: float = -1.5
    age_penalty: float = 0.0  # Pure content addressing by default!
    # Group B posterior fusion temperature & family priors
    b_temp: float = 1.0
    b_cfo_zero_bonus: float = 1.2
    b_quad_zero_bonus: float = 0.0  # Uniform over 4 quadrants in prior (let pilots & history resolve!)
    b_var_scale: float = 1.0        # Variance calibration multiplier tuned on dev


def estimate_train_channel_prior(y_sym_train: np.ndarray) -> tuple[complex, float]:
    """Estimate empirical (m0, v0) from train split using ONLY online pilots [13:21] (zero ground truth!)."""
    s_eff = S_PIL + PRIOR_DC_RATIO
    denom = float(np.sum(np.abs(s_eff) ** 2))
    h_pils = (y_sym_train[:, PILOT_SLICE] @ np.conj(s_eff)) / denom
    m0 = complex(np.mean(h_pils))
    v0 = float(np.mean(np.abs(h_pils - m0) ** 2))
    # Add generous safety margin so prior stays well-calibrated across fading regimes
    return m0, max(v0 * 1.5, 1e-6)


def estimate_single_frame_pilot_stats(
    y_pil: np.ndarray,
    h_prior_mean: complex,
    h_prior_var: float,
) -> dict[str, Any]:
    """Extract pilot LS/LMMSE channel, DC offset, and robust noise variance from 8 online pilots [13:21]."""
    # Effective zero-DC-adjusted pilot vector s_eff = s_pil + PRIOR_DC_RATIO
    s_eff = S_PIL + PRIOR_DC_RATIO
    se_en = float(np.sum(np.abs(s_eff) ** 2))

    # 1. Naive 1-tap LS (no DC)
    h_ls_1tap = complex(np.vdot(S_PIL, y_pil) / S_PIL_EN)
    res_1tap = y_pil - h_ls_1tap * S_PIL
    sig2_1tap = float(np.sum(np.abs(res_1tap) ** 2) / max(N_PIL - 1, 1))

    # 2. Unregularized 2-param (h, d) LS on 8 pilots
    w_ls_dc, _, _, _ = np.linalg.lstsq(X_PIL_DC, y_pil, rcond=None)
    h_ls_dc = complex(w_ls_dc[0])
    d_ls_dc = complex(w_ls_dc[1])
    res_dc = y_pil - (h_ls_dc * S_PIL + d_ls_dc)
    sig2_dc = float(np.sum(np.abs(res_dc) ** 2) / max(N_PIL - 2, 1))

    # 3. Physics-regularized AC-coupled model: y_p = h * s_eff + delta_d + w
    # Using s_eff absorbs the deterministic mean Barker DC while allowing small random payload-mean delta_d!
    h_ac_ls = complex(np.vdot(s_eff, y_pil) / se_en)
    res_ac = y_pil - h_ac_ls * s_eff
    # Robust noise variance estimate: combine pilot residual with lower floor from hardware ISI floor (~0.015 * |h|^2)
    h_pow = max(abs(h_ac_ls) ** 2, abs(h_prior_mean) ** 2 * 0.1, 1e-10)
    raw_var = float(np.sum(np.abs(res_ac) ** 2) / max(N_PIL - 1, 1))
    # Shrink noisy 7-dof variance estimate slightly toward realistic floor to avoid catastrophic under-estimation
    sigma2_est = max(raw_var, 0.015 * h_pow)

    # Conjugate Gaussian LMMSE update for h under prior CN(h_prior_mean, h_prior_var)
    prec_prior = 1.0 / max(h_prior_var, 1e-12)
    prec_lik = se_en / sigma2_est
    post_var_h = 1.0 / (prec_prior + prec_lik)
    h_mmse = post_var_h * (prec_prior * h_prior_mean + (np.vdot(s_eff, y_pil) / sigma2_est))

    # Empirical Bayes DC update around h_mmse * PRIOR_DC_RATIO
    d_prior_mean = h_mmse * PRIOR_DC_RATIO
    d_prior_var = max(abs(h_mmse) ** 2 * PRIOR_DC_REL_VAR, 1e-10)
    res_after_h = y_pil - (h_mmse * S_PIL + d_prior_mean)
    prec_d_prior = 1.0 / d_prior_var
    prec_d_lik = float(N_PIL) / sigma2_est
    post_var_d = 1.0 / (prec_d_prior + prec_d_lik)
    delta_d = post_var_d * (np.sum(res_after_h) / sigma2_est)
    d_mmse = d_prior_mean + delta_d

    # Refined residual variance including small uncertainty
    res_mmse = y_pil - (h_mmse * S_PIL + d_mmse)
    sigma2_ref = max(float(np.mean(np.abs(res_mmse) ** 2)) * (N_PIL / max(N_PIL - 1.5, 1.0)), 0.015 * max(abs(h_mmse) ** 2, 1e-10))

    return {
        "h_ls_1tap": h_ls_1tap,
        "sig2_1tap": max(sig2_1tap, 1e-10),
        "h_ls_dc": h_ls_dc,
        "d_ls_dc": d_ls_dc,
        "sig2_dc": max(sig2_dc, 1e-10),
        "h_ac_ls": h_ac_ls,
        "h_mmse": complex(h_mmse),
        "d_mmse": complex(d_mmse),
        "var_h_mmse": float(post_var_h),
        "var_d_mmse": float(post_var_d),
        "sigma2_est": float(sigma2_ref),
    }


def viterbi_viterbi_cfo_est(y_pay: np.ndarray, d_est: complex) -> float:
    """Classical 4th-power Viterbi-Viterbi blind CFO estimator on QPSK payload (covers [-0.04, +0.04] rad/sym)."""
    z = y_pay - d_est
    # Normalize amplitude lightly to prevent outliers from dominating 4th power
    amp = np.abs(z) + 1e-12
    z4 = (z / np.sqrt(amp)) ** 4  # M-th power with soft normalization
    # Fine grid search over [-0.038, +0.038] rad/sym (step 0.001 -> 77 points)
    omegas = np.linspace(-0.038, 0.038, 77, dtype=np.float64)
    k = np.arange(len(z), dtype=np.float64)
    spec = np.abs(np.exp(-1j * 4.0 * omegas[:, None] * k[None, :]) @ z4)
    idx = int(np.argmax(spec))
    # Parabolic refinement
    if 0 < idx < len(omegas) - 1:
        yl, y0, yr = float(spec[idx - 1]), float(spec[idx]), float(spec[idx + 1])
        den = yl - 2.0 * y0 + yr
        frac = float(np.clip(0.5 * (yl - yr) / den, -0.5, 0.5)) if abs(den) > 1e-12 else 0.0
        step = omegas[1] - omegas[0]
        return float(omegas[idx] + frac * step)
    return float(omegas[idx])


def build_causal_history_library(
    y_sym_seq: np.ndarray,
    h_mean_prior: complex,
    h_var_prior: float,
    lib_size: int = 200,
) -> dict[str, np.ndarray]:
    """Build strictly causal, ZERO-LEAKAGE blind soft-decision-directed channel states for all frames.

    For each frame j, estimates (h_dd[j], d_dd[j], var_h_dd[j], sigma2_dd[j], cfo_dd[j], quality[j])
    using ONLY online pilots y_j[13:21] + high-confidence soft-demodulated payload y_j[26:126].
    NEVER touches held-out probe y_j[21:26] or ground-truth Tx_data!
    All downstream arms (A, B, C) read exclusively from causal past indices [t-lib_size : t].
    """
    n = len(y_sym_seq)
    h_dd = np.empty(n, dtype=np.complex128)
    d_dd = np.empty(n, dtype=np.complex128)
    var_h_dd = np.empty(n, dtype=np.float64)
    sigma2_dd = np.empty(n, dtype=np.float64)
    cfo_dd = np.empty(n, dtype=np.float64)
    qual_dd = np.empty(n, dtype=np.float64)

    for j in range(n):
        yj = np.asarray(y_sym_seq[j], dtype=np.complex128)
        st = estimate_single_frame_pilot_stats(yj[PILOT_SLICE], h_mean_prior, h_var_prior)
        h0, d0, sig2 = st["h_mmse"], st["d_mmse"], st["sigma2_est"]

        # Quick blind CFO check via 4-point grid / VV on payload to prevent CFO-corrupted entries
        # Use pilot + payload soft-EM 1-step refinement over 7 CFO grid values
        y_pay = yj[PAYLOAD_SLICE]
        best_ll = -1e30
        best_cfo = 0.0
        best_h = h0
        best_d = d0
        best_s_soft = None
        best_w_soft = None

        for g_idx, w_g in enumerate(CFO_GRID):
            rot_p = np.exp(1j * w_g * (K_PIL - K_CENTER))
            s_eff_g = S_PIL * rot_p + PRIOR_DC_RATIO
            en_g = float(np.sum(np.abs(s_eff_g) ** 2))
            prec_p = 1.0 / max(h_var_prior, 1e-12)
            v_g = 1.0 / (prec_p + en_g / sig2)
            h_g = v_g * (prec_p * h_mean_prior + np.vdot(s_eff_g, yj[PILOT_SLICE]) / sig2)
            d_g = h_g * PRIOR_DC_RATIO

            # Payload marginal log-likelihood under QPSK constellation (100% blind to truth!)
            h_pay_g = h_g * np.exp(1j * w_g * (K_PAY - K_CENTER))
            pred = h_pay_g[:, None] * QPSK_CONST[None, :] + d_g  # (100, 4)
            v_obs = sig2 + v_g * 1.0
            d2 = (y_pay[:, None].real - pred.real) ** 2 + (y_pay[:, None].imag - pred.imag) ** 2
            logp = -d2 / v_obs - math.log(math.pi * v_obs) - math.log(4.0)
            lse = logsumexp(logp, axis=1)
            # Pilot log-likelihood + weak zero-CFO prior
            res_p = yj[PILOT_SLICE] - (h_g * S_PIL * rot_p + d_g)
            ll_pil = -float(np.sum(np.abs(res_p) ** 2)) / v_obs
            ll_tot = float(np.sum(lse)) + ll_pil + (1.0 if abs(w_g) < 1e-9 else 0.0)
            if ll_tot > best_ll:
                best_ll = ll_tot
                best_cfo = float(w_g)
                best_h = complex(h_g)
                best_d = complex(d_g)
                probs = np.exp(logp - lse[:, None])
                best_s_soft = probs @ QPSK_CONST
                best_w_soft = np.clip((np.abs(best_s_soft) ** 2) / 0.9997, 0.0, 1.0)

        # Weighted soft-DD refinement combining 8 known pilots (weight 1.0) + confident payload (weight w^2)
        rot_pil = np.exp(1j * best_cfo * (K_PIL - K_CENTER))
        rot_pay = np.exp(1j * best_cfo * (K_PAY - K_CENTER))
        w_pay = np.where(best_w_soft > 0.35, best_w_soft ** 2, 0.0)
        n_eff_pay = float(np.sum(w_pay))

        if n_eff_pay > 5.0:
            # Joint weighted LS on [s*rot, 1] over pilots [13:21] and soft payload [26:126] (NEVER probe [21:26]!)
            s_comb = np.concatenate([S_PIL * rot_pil, best_s_soft * rot_pay])
            y_comb = np.concatenate([yj[PILOT_SLICE], y_pay])
            wt = np.concatenate([np.ones(N_PIL, dtype=np.float64), w_pay])
            sw = np.sqrt(wt)
            X = np.column_stack([s_comb * sw, sw])
            yw = y_comb * sw
            # Tiny ridge towards pilot prior for complete numerical stability
            XtX = X.conj().T @ X
            Xty = X.conj().T @ yw
            reg = np.diag([sig2 / max(h_var_prior, 1e-10), sig2 / max(abs(best_h) ** 2 * PRIOR_DC_REL_VAR, 1e-10)])
            rhs_prior = np.array([h_mean_prior * reg[0, 0], (best_h * PRIOR_DC_RATIO) * reg[1, 1]], dtype=np.complex128)
            w_dd = np.linalg.solve(XtX + reg, Xty + rhs_prior)
            hj, dj = complex(w_dd[0]), complex(w_dd[1])
            res_comb = y_comb - (hj * s_comb + dj)
            sig2_j = max(float(np.sum(wt * (np.abs(res_comb) ** 2)) / max(float(np.sum(wt)) - 2.0, 1.0)), 0.012 * max(abs(hj) ** 2, 1e-10))
            eff_n = float(N_PIL + 0.7 * n_eff_pay)
            var_hj = sig2_j / max(eff_n, 1.0)
            qual_j = float(n_eff_pay / 100.0)
        else:
            hj, dj, sig2_j = best_h, best_d, sig2
            var_hj = st["var_h_mmse"]
            qual_j = 0.0

        h_dd[j] = hj
        d_dd[j] = dj
        var_h_dd[j] = var_hj
        sigma2_dd[j] = sig2_j
        cfo_dd[j] = best_cfo
        qual_dd[j] = qual_j

    return {
        "h_dd": h_dd,
        "d_dd": d_dd,
        "var_h_dd": var_h_dd,
        "sigma2_dd": sigma2_dd,
        "cfo_dd": cfo_dd,
        "qual_dd": qual_dd,
    }


def _fuse_prior_with_pilots(
    y_pil: np.ndarray,
    h_prior: complex,
    var_h_prior: float,
    d_prior: complex,
    sigma2: float,
    rot_pil: np.ndarray | None = None,
) -> tuple[complex, complex, float, float, float]:
    """Conjugate Gaussian fusion of prior CN(h_prior, var_h_prior) with 8 online pilots.

    Returns (h_post, d_post, var_h_post, var_d_post, marginal_log_evidence).
    """
    sp = S_PIL if rot_pil is None else (S_PIL * rot_pil)
    # Center DC around physics-coupled prior d_prior
    y_adj = y_pil - d_prior
    sp_en = float(np.sum(np.abs(sp) ** 2))
    v_pr = max(float(var_h_prior), 1e-12)
    s2 = max(float(sigma2), 1e-12)

    # Exact marginal log-evidence log N(y_adj; h_prior * sp, s2*I + v_pr * sp sp^H) via matrix determinant lemma!
    # Sigma = s2 I + v_pr sp sp^H
    # det(Sigma) = s2^N * (1 + (v_pr / s2) * ||sp||^2)
    # r^H Sigma^{-1} r = (1/s2) ||r||^2 - (v_pr / (s2 * (s2 + v_pr * ||sp||^2))) |sp^H r|^2
    r = y_adj - h_prior * sp
    r2 = float(np.sum(r.real ** 2 + r.imag ** 2))
    sp_r = np.vdot(sp, r)
    denom = s2 + v_pr * sp_en
    log_det = N_PIL * math.log(math.pi * s2) + math.log(denom / s2)
    quad = (r2 / s2) - (v_pr / (s2 * denom)) * float(sp_r.real ** 2 + sp_r.imag ** 2)
    log_ev = -(log_det + quad)

    # Posterior for h
    prec_pr = 1.0 / v_pr
    prec_lk = sp_en / s2
    var_h_post = 1.0 / (prec_pr + prec_lk)
    h_post = var_h_post * (prec_pr * h_prior + np.vdot(sp, y_adj) / s2)

    # Empirical Bayes tiny refinement of DC offset around d_prior
    d_rel_var = max(abs(h_post) ** 2 * PRIOR_DC_REL_VAR, 1e-10)
    res_d = y_pil - (h_post * sp + d_prior)
    var_d_post = 1.0 / (1.0 / d_rel_var + float(N_PIL) / s2)
    d_post = d_prior + var_d_post * (np.sum(res_d) / s2)

    return complex(h_post), complex(d_post), float(var_h_post), float(var_d_post), float(log_ev)


def compute_content_prior(
    y_pil: np.ndarray,
    lib_h: np.ndarray,
    lib_d: np.ndarray,
    lib_vh: np.ndarray,
    sigma2: float,
    hp: ReceiverHyperparams,
) -> dict[str, Any]:
    """EM-FloatFormer physical content-addressing attention over causal library M_t + self/clear branch."""
    m = len(lib_h)
    # Predictive variance for each library entry evaluating current pilots
    h_scale2 = np.maximum(np.abs(lib_h) ** 2, 1e-10)
    nu = sigma2 + lib_vh + hp.attn_drift_rel_var * h_scale2  # (M,)
    # Residual ||y_pil - (h_j * S_PIL + d_j)||^2 for all M entries vectorized!
    pred = lib_h[:, None] * S_PIL[None, :] + lib_d[:, None]  # (M, 8)
    diff = y_pil[None, :] - pred
    res2 = np.sum(diff.real ** 2 + diff.imag ** 2, axis=1)  # (M,)
    ll_lib = -(N_PIL * np.log(np.pi * nu) + res2 / nu)
    if hp.age_penalty > 0.0:
        ages = np.arange(m, 0, -1, dtype=np.float64) / float(m)
        ll_lib = ll_lib - hp.age_penalty * ages

    # Self/clear branch marginal log-evidence under broad empirical prior
    _, _, _, _, ll_clear = _fuse_prior_with_pilots(
        y_pil, hp.h_mean_prior, hp.h_var_prior, hp.h_mean_prior * PRIOR_DC_RATIO, sigma2
    )
    scores = np.empty(m + 1, dtype=np.float64)
    scores[:m] = (ll_lib - math.log(float(m))) / max(hp.attn_temp, 1e-3)
    scores[m] = (ll_clear + hp.clear_prior_log_bias) / max(hp.attn_temp, 1e-3)
    lse = float(logsumexp(scores))
    w = np.exp(scores - lse)

    # Mixture prior moments (preserving BOTH within-branch and between-branch variance!)
    w_lib = w[:m]
    w_clr = float(w[m])
    h_all = np.append(lib_h, hp.h_mean_prior)
    d_all = np.append(lib_d, hp.h_mean_prior * PRIOR_DC_RATIO)
    v_all = np.append(lib_vh + hp.attn_drift_rel_var * h_scale2, hp.h_var_prior)

    h_mix = complex(np.sum(w * h_all))
    d_mix = complex(np.sum(w * d_all))
    # Law of total variance (posterior_fusion §4.1!)
    v_within = float(np.sum(w * v_all))
    v_between = float(np.sum(w * (np.abs(h_all - h_mix) ** 2)))
    v_tot = max(v_within + v_between, 1e-10)
    return {
        "h_prior": h_mix,
        "d_prior": d_mix,
        "var_h_prior": v_tot,
        "weights": w,
        "clear_weight": w_clr,
        "eff_n_refs": float(1.0 / max(float(np.sum(w ** 2)), 1e-12)),
    }


def build_shared_b_candidates(
    y_pil: np.ndarray,
    sigma2: float,
    priors_3: list[tuple[complex, float, complex]],  # [self_clear, local_neighbor, content_global]
    hp: ReceiverHyperparams,
    extra_logit_bias: np.ndarray | None = None,
) -> dict[str, Any]:
    """Construct the exact shared 84-branch physical candidate bank (4 quad x 7 CFO x 3 prior families).

    Computes exact pilot-conditioned posterior (h_r, d_r, var_h_r, var_d_r) and marginal log-evidence
    for every branch r in 0..83 using ONLY online pilots y_pil and causal priors.
    """
    n_q = len(QUAD_PHASES)
    n_g = len(CFO_GRID)
    n_c = len(priors_3)
    r_tot = n_q * n_g * n_c

    h_post = np.empty(r_tot, dtype=np.complex128)
    d_post = np.empty(r_tot, dtype=np.complex128)
    var_h_post = np.empty(r_tot, dtype=np.float64)
    var_d_post = np.empty(r_tot, dtype=np.float64)
    log_ev = np.empty(r_tot, dtype=np.float64)
    log_pi = np.empty(r_tot, dtype=np.float64)
    q_idx_arr = np.empty(r_tot, dtype=np.int32)
    g_idx_arr = np.empty(r_tot, dtype=np.int32)
    c_idx_arr = np.empty(r_tot, dtype=np.int32)

    idx = 0
    for c_i, (h_pr, v_pr, d_pr) in enumerate(priors_3):
        for q_i in range(n_q):
            # Quadrant rotation applies to the reference prior phase hypothesis:
            # rotating the channel hypothesis by e^{j*q*pi/2} tests all 4 QPSK phase ambiguity states!
            rot_q = np.exp(1j * QUAD_PHASES[q_i])
            h_pr_q = h_pr * rot_q
            d_pr_q = d_pr * rot_q
            for g_i in range(n_g):
                rot_g_pil = np.exp(1j * CFO_GRID[g_i] * (K_PIL - K_CENTER))
                hp_r, dp_r, vhp_r, vdp_r, lev = _fuse_prior_with_pilots(
                    y_pil, h_pr_q, v_pr, d_pr_q, sigma2, rot_pil=rot_g_pil
                )
                h_post[idx] = hp_r
                d_post[idx] = dp_r
                var_h_post[idx] = vhp_r
                var_d_post[idx] = vdp_r
                log_ev[idx] = lev
                # Prospective prior: slight bonus for 0-CFO (parsimony), uniform over 3 families
                lp = (hp.b_cfo_zero_bonus if g_i == 3 else 0.0) + (hp.b_quad_zero_bonus if q_i == 0 else 0.0)
                log_pi[idx] = lp
                q_idx_arr[idx] = q_i
                g_idx_arr[idx] = g_i
                c_idx_arr[idx] = c_i
                idx += 1

    logits = (log_pi + log_ev) / max(hp.b_temp, 1e-3)
    if extra_logit_bias is not None:
        logits = logits + extra_logit_bias
    alpha = np.exp(logits - float(logsumexp(logits)))

    # Build per-symbol predicted channel h_all[r, k] = h_post[r] * exp(j * omega_g[r] * (k - K_CENTER))
    # Note: the quadrant phase q_i is ALREADY absorbed into h_post[r] via h_pr_q!
    cfo_per_r = CFO_GRID[g_idx_arr]
    rot_all = np.exp(1j * cfo_per_r[:, None] * (K_ALL[None, :] - K_CENTER))  # (84, 126)
    h_all = h_post[:, None] * rot_all                                       # (84, 126)
    # Predictive observation variance per branch (at symbol energy ~1.0)
    obs_var_r = (sigma2 + var_h_post + var_d_post) * hp.b_var_scale         # (84,)

    return {
        "alpha": alpha,
        "logits_raw": log_pi + log_ev,
        "h_post": h_post,
        "d_post": d_post,
        "var_h_post": var_h_post,
        "var_d_post": var_d_post,
        "h_all": h_all,
        "obs_var_r": obs_var_r,
        "q_idx": q_idx_arr,
        "g_idx": g_idx_arr,
        "c_idx": c_idx_arr,
    }


def decode_group_b_variants(
    y_sym: np.ndarray,
    bank: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Run all 4 Group B decision rules on the EXACT same 84-branch candidate bank & alpha_r!

    Variants:
      - B1_hard_map: single argmax_r alpha_r branch
      - B2_posterior_mean_plugin: posterior mean channel + total variance (within + between) plug-in QPSK
      - B3a_sym_mixture: exact per-symbol mixture marginalization over all 84 branches using alpha_r
      - B3b_joint_mixture: exact full-frame payload mixture posterior beta_r (posterior_fusion §5.2)
                           marginalized over all 84 branches for symbol-MAP and bit-LLR-MAP decisions!
    """
    alpha = bank["alpha"]       # (84,)
    h_all = bank["h_all"]       # (84, 126)
    d_post = bank["d_post"]     # (84,)
    obs_var = bank["obs_var_r"] # (84,)
    y_pay = y_sym[PAYLOAD_SLICE]  # (100,)
    h_pay = h_all[:, PAYLOAD_SLICE]  # (84, 100)

    # --- B1: Hard MAP ---
    t0 = time.perf_counter()
    r_hat = int(np.argmax(alpha))
    h_b1 = h_all[r_hat]
    d_b1 = complex(d_post[r_hat])
    v_b1 = np.full(N_SYM, float(obs_var[r_hat]), dtype=np.float64)
    eq_b1 = (y_pay - d_b1) / np.where(np.abs(h_b1[PAYLOAD_SLICE]) > 1e-12, h_b1[PAYLOAD_SLICE], 1e-12)
    sdec_b1, idx_b1 = nearest_qpsk_symbols(eq_b1)
    bits_b1 = QPSK_BITS[idx_b1]
    t_b1 = (time.perf_counter() - t0) * 1000.0

    # --- B2: Posterior Mean Plug-in (with Law of Total Variance!) ---
    t0 = time.perf_counter()
    h_b2 = np.sum(alpha[:, None] * h_all, axis=0)                  # (126,)
    d_b2 = complex(np.sum(alpha * d_post))
    diff_h = h_all - h_b2[None, :]                                 # (84, 126)
    diff_d = d_post - d_b2                                         # (84,)
    v_between = np.sum(alpha[:, None] * (diff_h.real ** 2 + diff_h.imag ** 2), axis=0) + float(
        np.sum(alpha * (diff_d.real ** 2 + diff_d.imag ** 2))
    )
    v_within = float(np.sum(alpha * obs_var))
    v_b2 = v_within + v_between                                    # (126,)
    eq_b2 = (y_pay - d_b2) / np.where(np.abs(h_b2[PAYLOAD_SLICE]) > 1e-12, h_b2[PAYLOAD_SLICE], 1e-12)
    sdec_b2, idx_b2 = nearest_qpsk_symbols(eq_b2)
    bits_b2 = QPSK_BITS[idx_b2]
    t_b2 = (time.perf_counter() - t0) * 1000.0

    # Shared exact branch-symbol likelihood tensor: log p(y_k, S_k=s | I, r) for (84, 100, 4)
    t_mix0 = time.perf_counter()
    # pred: (84, 100, 4)
    pred = h_pay[:, :, None] * QPSK_CONST[None, None, :] + d_post[:, None, None]
    err = y_pay[None, :, None] - pred
    d2 = err.real ** 2 + err.imag ** 2
    v_r = obs_var[:, None, None]
    log_p_y_s_given_r = -d2 / v_r - np.log(np.pi * v_r) - math.log(4.0)  # (84, 100, 4)
    log_alpha = np.log(np.maximum(alpha, 1e-300))                        # (84,)
    t_shared_mix = (time.perf_counter() - t_mix0) * 1000.0

    # --- B3a: Per-symbol Exact Mixture Likelihood (prior alpha_r only) ---
    t0 = time.perf_counter()
    log_joint_sym = log_alpha[:, None, None] + log_p_y_s_given_r         # (84, 100, 4)
    log_marg_s_a = logsumexp(log_joint_sym, axis=0)                      # (100, 4)
    p_s_a = np.exp(log_marg_s_a - logsumexp(log_marg_s_a, axis=1, keepdims=True))
    idx_b3a = np.argmax(p_s_a, axis=1)
    sdec_b3a = QPSK_CONST[idx_b3a]
    # Exact bit-wise MAP from marginal bit probabilities (b0 in {2,3}, b1 in {1,3})
    p_b0_a = p_s_a[:, 2] + p_s_a[:, 3]
    p_b1_a = p_s_a[:, 1] + p_s_a[:, 3]
    bits_b3a = np.stack([(p_b0_a > 0.5).astype(np.int8), (p_b1_a > 0.5).astype(np.int8)], axis=-1)
    # Soft MMSE symbol reconstruction E[S_k | I, y_k]
    eq_b3a = p_s_a @ QPSK_CONST
    t_b3a = t_shared_mix + (time.perf_counter() - t0) * 1000.0

    # --- B3b: Exact Full-Frame Joint Mixture Posterior (posterior_fusion §5.2!) ---
    t0 = time.perf_counter()
    # Exact per-symbol marginal log p(y_k | I, r) = logsumexp_s log p(y_k, S_k=s | I, r)
    log_p_y_given_r = logsumexp(log_p_y_s_given_r, axis=2)               # (84, 100)
    # Full-frame payload marginal log-likelihood per branch
    log_beta_unnorm = log_alpha + np.sum(log_p_y_given_r, axis=1)        # (84,)
    beta = np.exp(log_beta_unnorm - float(logsumexp(log_beta_unnorm)))   # (84,)
    # Exact conditional symbol posterior within each branch: P(S_k=s | I, y_k, r)
    p_s_given_r = np.exp(log_p_y_s_given_r - log_p_y_given_r[:, :, None])  # (84, 100, 4)
    # Exact mixture marginalization (ZERO double counting of y_k!): P(S_k=s | O) = sum_r beta_r P(S_k=s | y_k, r)
    p_s_b = np.tensordot(beta, p_s_given_r, axes=([0], [0]))             # (100, 4)
    idx_b3b = np.argmax(p_s_b, axis=1)
    sdec_b3b = QPSK_CONST[idx_b3b]
    p_b0_b = p_s_b[:, 2] + p_s_b[:, 3]
    p_b1_b = p_s_b[:, 1] + p_s_b[:, 3]
    bits_b3b = np.stack([(p_b0_b > 0.5).astype(np.int8), (p_b1_b > 0.5).astype(np.int8)], axis=-1)
    # Posterior-weighted equalized symbol (MMSE across beta-weighted branches, linear equalization for fair EVM!)
    eq_branch = (y_pay[None, :] - d_post[:, None]) / np.where(np.abs(h_pay) > 1e-12, h_pay, 1e-12)
    eq_b3b = np.sum(beta[:, None] * eq_branch, axis=0)
    t_b3b = t_shared_mix + (time.perf_counter() - t0) * 1000.0

    # Notice: for held-out probe [21:26] prediction, B3a and B3b use the EXACT pre-payload posterior mean (h_b2, d_b2, v_b2)
    # so probe evaluation NEVER leaks payload observations or probe observations!
    return {
        "B1_hard_map": {
            "h_all": h_b1, "d_pred": d_b1, "v_all": v_b1,
            "s_dec": sdec_b1, "bits_dec": bits_b1, "s_eq": eq_b1, "extra_ms": t_b1,
        },
        "B2_posterior_mean_plugin": {
            "h_all": h_b2, "d_pred": d_b2, "v_all": v_b2,
            "s_dec": sdec_b2, "bits_dec": bits_b2, "s_eq": eq_b2, "extra_ms": t_b2,
        },
        "B3a_sym_mixture": {
            "h_all": h_b2, "d_pred": d_b2, "v_all": v_b2,
            "s_dec": sdec_b3a, "bits_dec": bits_b3a, "s_eq": eq_b2, "extra_ms": t_b3a,
        },
        "B3b_joint_mixture": {
            "h_all": h_b2, "d_pred": d_b2, "v_all": v_b2,
            "s_dec": sdec_b3b, "bits_dec": bits_b3b, "s_eq": eq_b3b, "extra_ms": t_b3b,
            "beta": beta,
        },
    }


def run_all_classical_and_ab_arms(
    bundle: Mapping[str, np.ndarray],
    tx_mm: np.ndarray,
    hp: ReceiverHyperparams,
    lib_size: int = 200,
    cached_lib: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, np.ndarray]]:
    """Run all Classical Baselines, Group A (5 arms), and Group B (4 arms) paired on one split-condition bundle."""
    y_all = np.asarray(bundle["y_sym"], dtype=np.complex128)
    frame_indices = np.asarray(bundle["frame_indices"], dtype=np.int32)
    eval_offset = int(bundle["eval_offset"])
    h_ora_all = np.asarray(bundle["h_ora"], dtype=np.complex128)
    d_ora_all = np.asarray(bundle["d_ora"], dtype=np.complex128)
    sync_ok_all = np.asarray(bundle["sync_ok"], dtype=np.bool_)

    if cached_lib is None:
        lib = build_causal_history_library(y_all, hp.h_mean_prior, hp.h_var_prior, lib_size=lib_size)
    else:
        lib = cached_lib

    arm_names = [
        "base_ls_1tap",
        "base_ls_dc",
        "base_mmse_dc_cfo",
        "base_full20_ls_dc",
        "A1_local_single",
        "A2_neighbor_k5",
        "A3_global_uniform",
        "A4_global_reliability",
        "A5_global_content_attn",
        "B1_hard_map",
        "B2_posterior_mean_plugin",
        "B3a_sym_mixture",
        "B3b_joint_mixture",
    ]
    records: dict[str, list[dict[str, Any]]] = {name: [] for name in arm_names}

    # Precompute neighbor decay weights for k=5
    nb_k = hp.neighbor_k
    nb_w_raw = np.exp(-hp.neighbor_decay * np.arange(nb_k, 0, -1, dtype=np.float64))
    nb_w_raw /= np.sum(nb_w_raw)

    n_tot = len(y_all)
    for t in range(eval_offset, n_tot):
        fidx = int(frame_indices[t])
        y = y_all[t]
        y_pil = y[PILOT_SLICE]
        s_true = np.asarray(tx_mm[fidx], dtype=np.complex128)
        h_ora = complex(h_ora_all[t])
        d_ora = complex(d_ora_all[t])
        sync_ok = bool(sync_ok_all[t])

        # Strictly causal library slice [t - lib_size : t]
        l_start = max(0, t - lib_size)
        lib_h = lib["h_dd"][l_start:t]
        lib_d = lib["d_dd"][l_start:t]
        lib_vh = lib["var_h_dd"][l_start:t]

        # --- Common single-frame pilot statistics ---
        t0 = time.perf_counter()
        st = estimate_single_frame_pilot_stats(y_pil, hp.h_mean_prior, hp.h_var_prior)
        t_st = (time.perf_counter() - t0) * 1000.0
        sig2 = st["sigma2_est"]

        # 1. base_ls_1tap
        h1 = st["h_ls_1tap"]
        v1 = np.full(N_SYM, st["sig2_1tap"] * (1.0 + 1.0 / N_PIL), dtype=np.float64)
        records["base_ls_1tap"].append(evaluate_frame_record(
            fidx, y, s_true, h_ora, d_ora, sync_ok,
            np.full(N_SYM, h1, dtype=np.complex128), 0.0j, v1, latency_ms=t_st, n_candidates=1,
        ))

        # 2. base_ls_dc
        h_dc, d_dc = st["h_ls_dc"], st["d_ls_dc"]
        v_dc = np.full(N_SYM, st["sig2_dc"] * (1.0 + 2.0 / N_PIL), dtype=np.float64)
        records["base_ls_dc"].append(evaluate_frame_record(
            fidx, y, s_true, h_ora, d_ora, sync_ok,
            np.full(N_SYM, h_dc, dtype=np.complex128), d_dc, v_dc, latency_ms=t_st, n_candidates=1,
        ))

        # 3. base_mmse_dc_cfo (Mature classical single-frame receiver: LMMSE + DC + Viterbi-Viterbi blind CFO!)
        t0 = time.perf_counter()
        w_vv = viterbi_viterbi_cfo_est(y[PAYLOAD_SLICE], st["d_mmse"])
        rot_vv_pil = np.exp(1j * w_vv * (K_PIL - K_CENTER))
        h_vv, d_vv, vh_vv, vd_vv, _ = _fuse_prior_with_pilots(
            y_pil, hp.h_mean_prior, hp.h_var_prior, hp.h_mean_prior * PRIOR_DC_RATIO, sig2, rot_pil=rot_vv_pil
        )
        h_vv_all = h_vv * np.exp(1j * w_vv * (K_ALL - K_CENTER))
        v_vv_all = np.full(N_SYM, (sig2 + vh_vv + vd_vv) * hp.b_var_scale, dtype=np.float64)
        t_vv = t_st + (time.perf_counter() - t0) * 1000.0
        records["base_mmse_dc_cfo"].append(evaluate_frame_record(
            fidx, y, s_true, h_ora, d_ora, sync_ok,
            h_vv_all, d_vv, v_vv_all, latency_ms=t_vv, n_candidates=1,
        ))

        # 4. base_full20_ls_dc (Reference classical upper bound using [1:21], 20 symbols, still masking probe [21:26]!)
        t0 = time.perf_counter()
        w20, _, _, _ = np.linalg.lstsq(X_HDR20_DC, y[1:21], rcond=None)
        h20, d20 = complex(w20[0]), complex(w20[1])
        r20 = y[1:21] - (h20 * S_HDR20 + d20)
        s2_20 = max(float(np.sum(np.abs(r20) ** 2) / 18.0), 0.015 * max(abs(h20) ** 2, 1e-10))
        v20 = np.full(N_SYM, s2_20 * (1.0 + 2.0 / 20.0), dtype=np.float64)
        t_20 = (time.perf_counter() - t0) * 1000.0
        records["base_full20_ls_dc"].append(evaluate_frame_record(
            fidx, y, s_true, h_ora, d_ora, sync_ok,
            np.full(N_SYM, h20, dtype=np.complex128), d20, v20, latency_ms=t_20, n_candidates=1,
        ))

        # --- Group A: 5 strictly paired arms separating Reference Scope vs Content Addressing ---
        # A1: Local single frame LMMSE (M = 0)
        t0 = time.perf_counter()
        h_a1, d_a1, vh_a1, vd_a1, _ = _fuse_prior_with_pilots(
            y_pil, hp.h_mean_prior, hp.h_var_prior, hp.h_mean_prior * PRIOR_DC_RATIO, sig2
        )
        t_a1 = t_st + (time.perf_counter() - t0) * 1000.0
        records["A1_local_single"].append(evaluate_frame_record(
            fidx, y, s_true, h_ora, d_ora, sync_ok,
            np.full(N_SYM, h_a1, dtype=np.complex128), d_a1,
            np.full(N_SYM, (sig2 + vh_a1 + vd_a1) * hp.b_var_scale, dtype=np.float64),
            latency_ms=t_a1, n_candidates=1,
        ))

        # A2: Fixed local neighbor window (k = 5)
        t0 = time.perf_counter()
        nh = lib_h[-nb_k:]
        nd = lib_d[-nb_k:]
        nvh = lib_vh[-nb_k:]
        w_nb = nb_w_raw[-len(nh):]
        w_nb = w_nb / np.sum(w_nb)
        h_pr_nb = complex(np.sum(w_nb * nh))
        d_pr_nb = complex(np.sum(w_nb * nd))
        v_pr_nb = float(np.sum(w_nb * (nvh + np.abs(nh - h_pr_nb) ** 2 + hp.neighbor_drift_rel_var * np.abs(nh) ** 2)))
        h_a2, d_a2, vh_a2, vd_a2, _ = _fuse_prior_with_pilots(y_pil, h_pr_nb, v_pr_nb, d_pr_nb, sig2)
        t_a2 = t_st + (time.perf_counter() - t0) * 1000.0
        records["A2_neighbor_k5"].append(evaluate_frame_record(
            fidx, y, s_true, h_ora, d_ora, sync_ok,
            np.full(N_SYM, h_a2, dtype=np.complex128), d_a2,
            np.full(N_SYM, (sig2 + vh_a2 + vd_a2) * hp.b_var_scale, dtype=np.float64),
            latency_ms=t_a2, n_candidates=nb_k,
        ))

        # A3: Global uniform average (M = 200) -> isolates scope expansion WITHOUT content selection
        t0 = time.perf_counter()
        h_pr_u = complex(np.mean(lib_h))
        d_pr_u = complex(np.mean(lib_d))
        v_pr_u = float(np.mean(lib_vh + np.abs(lib_h - h_pr_u) ** 2 + hp.global_drift_rel_var * np.abs(lib_h) ** 2))
        h_a3, d_a3, vh_a3, vd_a3, _ = _fuse_prior_with_pilots(y_pil, h_pr_u, v_pr_u, d_pr_u, sig2)
        t_a3 = t_st + (time.perf_counter() - t0) * 1000.0
        records["A3_global_uniform"].append(evaluate_frame_record(
            fidx, y, s_true, h_ora, d_ora, sync_ok,
            np.full(N_SYM, h_a3, dtype=np.complex128), d_a3,
            np.full(N_SYM, (sig2 + vh_a3 + vd_a3) * hp.b_var_scale, dtype=np.float64),
            latency_ms=t_a3, n_candidates=len(lib_h),
        ))

        # A4: Global classical reliability weights (M = 200, inverse-variance only, content-blind)
        t0 = time.perf_counter()
        w_rel = 1.0 / (lib_vh + 1e-10)
        w_rel /= np.sum(w_rel)
        h_pr_r = complex(np.sum(w_rel * lib_h))
        d_pr_r = complex(np.sum(w_rel * lib_d))
        v_pr_r = float(np.sum(w_rel * (lib_vh + np.abs(lib_h - h_pr_r) ** 2 + hp.global_drift_rel_var * np.abs(lib_h) ** 2)))
        h_a4, d_a4, vh_a4, vd_a4, _ = _fuse_prior_with_pilots(y_pil, h_pr_r, v_pr_r, d_pr_r, sig2)
        t_a4 = t_st + (time.perf_counter() - t0) * 1000.0
        records["A4_global_reliability"].append(evaluate_frame_record(
            fidx, y, s_true, h_ora, d_ora, sync_ok,
            np.full(N_SYM, h_a4, dtype=np.complex128), d_a4,
            np.full(N_SYM, (sig2 + vh_a4 + vd_a4) * hp.b_var_scale, dtype=np.float64),
            latency_ms=t_a4, n_candidates=len(lib_h),
        ))

        # A5: EM-FloatFormer Physical Content-Addressing Attention (M = 200 + clear branch)
        t0 = time.perf_counter()
        ca = compute_content_prior(y_pil, lib_h, lib_d, lib_vh, sig2, hp)
        h_a5, d_a5, vh_a5, vd_a5, _ = _fuse_prior_with_pilots(
            y_pil, ca["h_prior"], ca["var_h_prior"], ca["d_prior"], sig2
        )
        t_a5 = t_st + (time.perf_counter() - t0) * 1000.0
        records["A5_global_content_attn"].append(evaluate_frame_record(
            fidx, y, s_true, h_ora, d_ora, sync_ok,
            np.full(N_SYM, h_a5, dtype=np.complex128), d_a5,
            np.full(N_SYM, (sig2 + vh_a5 + vd_a5) * hp.b_var_scale, dtype=np.float64),
            latency_ms=t_a5, n_candidates=len(lib_h) + 1,
        ))

        # --- Group B: Shared 84-branch Physical Candidate Bank (4 quad x 7 CFO x 3 prior families) ---
        t_b0 = time.perf_counter()
        priors_3 = [
            (hp.h_mean_prior, hp.h_var_prior, hp.h_mean_prior * PRIOR_DC_RATIO),
            (h_pr_nb, v_pr_nb, d_pr_nb),
            (ca["h_prior"], ca["var_h_prior"], ca["d_prior"]),
        ]
        bank = build_shared_b_candidates(y_pil, sig2, priors_3, hp)
        t_bank = t_a5 + (time.perf_counter() - t_b0) * 1000.0
        b_out = decode_group_b_variants(y, bank)

        for bname in ("B1_hard_map", "B2_posterior_mean_plugin", "B3a_sym_mixture", "B3b_joint_mixture"):
            bo = b_out[bname]
            records[bname].append(evaluate_frame_record(
                fidx, y, s_true, h_ora, d_ora, sync_ok,
                bo["h_all"], bo["d_pred"], bo["v_all"],
                s_dec_payload=bo["s_dec"],
                bits_dec_payload=bo["bits_dec"],
                s_eq_payload=bo["s_eq"],
                latency_ms=t_bank + bo["extra_ms"],
                n_candidates=N_CANDIDATES_B,
            ))

    return records, lib
#!/usr/bin/env python3
"""Download and verify Mendeley Data 'Frames of SDR ADALM PLUTO based QPSK Communication Link'.

Dataset DOI: 10.17632/k72b4pnc5g.1
Official URL: https://data.mendeley.com/datasets/k72b4pnc5g/1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

DATASET_ID = "k72b4pnc5g"
DATASET_VERSION = "1"
METADATA_URL = f"https://data.mendeley.com/public-api/datasets/{DATASET_ID}?version={DATASET_VERSION}"
FILES_URL = f"https://data.mendeley.com/public-api/datasets/{DATASET_ID}/files?folder_id=root&version={DATASET_VERSION}"
EXPECTED_FILENAME = "QPSKlink.mat"
EXPECTED_SIZE = 822762822
EXPECTED_SHA256 = "c029532908f9702fea4be3ce37d0e1de9138019e12dd06e228abb1b83525f2b7"


def fetch_json(url: str, out_path: Path) -> dict | list:
    req = urllib.request.Request(url, headers={"User-Agent": "EM-FloatFormer-Research/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read()
    out_path.write_bytes(raw)
    return json.loads(raw.decode("utf-8"))


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            buf = f.read(chunk_size)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def download_stream(url: str, dest: Path, expected_size: int | None = None) -> tuple[int, str, float]:
    tmp = dest.with_suffix(dest.suffix + ".part")
    h = hashlib.sha256()
    total = 0
    t0 = time.perf_counter()
    req = urllib.request.Request(url, headers={"User-Agent": "EM-FloatFormer-Research/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as out:
        while True:
            chunk = resp.read(4 * 1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            h.update(chunk)
            total += len(chunk)
    elapsed = time.perf_counter() - t0
    if expected_size is not None and total != expected_size:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Size mismatch: got {total}, expected {expected_size}")
    os.replace(tmp, dest)
    return total, h.hexdigest(), elapsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True, help="Directory to store raw dataset and metadata")
    ap.add_argument("--manifest-out", type=Path, default=None, help="Optional path to write verification manifest JSON")
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_dir / "dataset_metadata.json"
    files_path = out_dir / "files_metadata.json"
    dataset_meta = fetch_json(METADATA_URL, meta_path)
    files_meta = fetch_json(FILES_URL, files_path)

    target_entry = None
    for entry in files_meta:
        if entry.get("filename") == EXPECTED_FILENAME:
            target_entry = entry
            break
    if target_entry is None:
        raise RuntimeError(f"Could not find {EXPECTED_FILENAME} in Mendeley files metadata")

    details = target_entry["content_details"]
    dl_url = details["download_url"]
    api_size = int(details["size"])
    api_sha256 = str(details["sha256_hash"]).lower()

    mat_path = out_dir / EXPECTED_FILENAME
    if mat_path.exists() and mat_path.stat().st_size == api_size:
        local_sha = sha256_file(mat_path)
        if local_sha == api_sha256:
            download_method = "cached_verified"
            elapsed_s = 0.0
            actual_size = api_size
        else:
            actual_size, local_sha, elapsed_s = download_stream(dl_url, mat_path, expected_size=api_size)
            download_method = "urllib_stream_redownload"
    else:
        actual_size, local_sha, elapsed_s = download_stream(dl_url, mat_path, expected_size=api_size)
        download_method = "urllib_stream"

    if local_sha != api_sha256 or local_sha != EXPECTED_SHA256:
        raise RuntimeError(f"SHA-256 mismatch: local={local_sha}, api={api_sha256}, expected={EXPECTED_SHA256}")

    with mat_path.open("rb") as f:
        header_bytes = f.read(128)
    is_hdf5 = header_bytes[:8] == b"\x89HDF\r\n\x1a\n"
    mat_header_ascii = header_bytes[:116].decode("latin1", errors="replace").strip("\x00 ")

    manifest = {
        "dataset_title": dataset_meta.get("name"),
        "dataset_doi": f"10.17632/{DATASET_ID}.{DATASET_VERSION}",
        "dataset_url": f"https://data.mendeley.com/datasets/{DATASET_ID}/{DATASET_VERSION}",
        "public_api_metadata_url": METADATA_URL,
        "public_api_files_url": FILES_URL,
        "download_url": dl_url,
        "download_method": download_method,
        "download_elapsed_sec": round(elapsed_s, 3),
        "filename": EXPECTED_FILENAME,
        "size_bytes": actual_size,
        "sha256": local_sha,
        "sha256_matches_mendeley_api": local_sha == api_sha256,
        "is_hdf5_v73": is_hdf5,
        "header_preview": mat_header_ascii[:100],
        "metadata_sha256": sha256_file(meta_path),
        "files_metadata_sha256": sha256_file(files_path),
        "verified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    manifest_path = args.manifest_out or (out_dir / "data_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

### Event #9: `item.completed` — `command_execution` (`item_5`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "Get-Content research/matched_cache/learned.py; rg -n 'def .*match|matched_sum|B3_mixture|RefPool' research/matched_cache research/stage3d_qwen_signed_phase -g '*.py'; Get-ChildItem -Force -Name AGENTS.md, research/AGENTS.md,research/stage4_math_and_design/AGENTS.md,research/stage4_math_and_design/M4/AGENTS.md -ErrorAction SilentlyContinue; git rev-parse HEAD origin/research/realdata-pluto-qpsk origin/research/realdata-cttc-indoor-qpsk; uv --version"
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
research/stage3d_qwen_signed_phase\data_stage3d.py:31:    matched_sum,
research/stage3d_qwen_signed_phase\analysis_stage3d.py:21:    matched_sum,
research/stage3d_qwen_signed_phase\analysis_stage3d.py:143:            targets[s_i, r_i] = sector_of_complex(matched_sum(batch.y3_repeat * ROOTS[s1 % 4])).astype(np.uint8)
research/stage3d_qwen_signed_phase\analysis_stage3d.py:144:            indep_targets[s_i, r_i] = sector_of_complex(matched_sum(batch.y3_indep * ROOTS[s1 % 4])).astype(np.uint8)
research/stage3d_qwen_signed_phase\analysis_stage3d.py:155:        rot10_targets[s_i] = sector_of_complex(matched_sum(batch.y3_repeat * ROOTS[1])).astype(np.uint8)
research/stage3d_qwen_signed_phase\analysis_stage3d.py:210:            targets[s_i, r_i] = sector_of_complex(matched_sum(batch.y3_repeat * ROOTS[s1 % 4])).astype(np.uint8)
research/stage3d_qwen_signed_phase\analysis_stage3d.py:398:            rec = [(r.acquisition_id, bool(r.shares_channel), complex(matched_sum(r.samples[row_i]))) for r in refs_tuple]
research/stage3d_qwen_signed_phase\analysis_stage3d.py:409:        q2__matched_valid1=matched_sum(q2_batch.p_valid1),
research/stage3d_qwen_signed_phase\analysis_stage3d.py:410:        q2__matched_valid2=matched_sum(q2_batch.p_valid2),
research/stage3d_qwen_signed_phase\analysis_stage3d.py:411:        q2__matched_indep=matched_sum(q2_batch.p_indep),
research/stage3d_qwen_signed_phase\analysis_stage3d.py:431:                lbl_q3.append(sector_of_complex(matched_sum(batch.y3_repeat * ROOTS[s1 % 4])))
research/stage3d_qwen_signed_phase\train_eval_stage3d.py:24:    matched_sum,
research/stage3d_qwen_signed_phase\train_eval_stage3d.py:119:                    "labels": sector_of_complex(matched_sum(y3_rot)),
research/stage3d_qwen_signed_phase\train_eval_stage3d.py:156:        y_q3_rep = sector_of_complex(matched_sum(batch.y3_repeat * ROOTS[s1 % 4]))
research/stage3d_qwen_signed_phase\tests\test_stage3d_cpu.py:12:from research.stage3_ministral_cache.schema import ROOTS, ROTATIONS, matched_sum, sector_of_complex
research/stage3d_qwen_signed_phase\tests\test_stage3d_cpu.py:78:    np.testing.assert_array_equal(sector_of_complex(matched_sum(pos.y3_repeat * ROOTS[1])), (pos.k3_repeat + 1) % 4)
387e7bc749150a69fb317b3f1351ad0d07b3bfad
ade8333fb68126251a83e484b69b821b5d97bcea
1887aee6de9649bc0dd8a4f05fe3c4542f3f6d77
uv 0.11.21 (5aa65dd7a 2026-06-11 x86_64-pc-windows-msvc)
```

### Event #10: `item.completed` — `command_execution` (`item_3`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command 'Get-Content docs/LONG_TERM_RESEARCH_PLAN.md; Get-Content research/stage3d_qwen_signed_phase/PROTOCOL_LOCK.json; Get-Content research/stage3d_qwen_signed_phase/REPORT.md'
```

**Aggregated Output:**

````text
# EM\-FloatFormer 长期研究计划

## �?AI 从同一段电磁波形中读出关系、指出依据，并随新证据更新答�?

**研究快照�?2026 �?11 �?22 日�?* 本计划覆�?GitHub 当时可见的主分支�?17 条研究分支。分支末端与报告列在文末。仓库审阅以各阶段报告、证明和实测数据说明为依据，本轮没有重训模型或重跑历史波形。时间按一位研究者约 6�?2 个月估计，阶段以完成的研究成果推进�?

### 总目标与双重验收

目标是让已有 AI 把一段民用无�?I/Q 录制作为原生感知输入。模型先读波形，保留一份大小明确、可回指时间窗的表示。用户随后才提出问题，模型可以描述关系、定位变化、指出歧义，并在得到一段额外参考后修订先前的回答。研究顺序是先给出可证伪的物理预测，再构造对照，随后训练，最后在独立采集上检验�?

**自己的起点�?* P7 已能预测留出的导频并更新判断，P10 已区分复相关保留的相对相位和消失的公共相位，P12 �?A–E 会合已研究重复计入证据、远端内容寻址和可辨识下界。下一步要证明或证伪，一份在提问前形成的有限表示，能够在何种信道条件下同时服务多类追问。第一项理论目标是给出缓存预算、参考间隔及噪声与逐题风险的关系，进而构造达到该界或接近它的表示。只有曲线的上界、下界或两侧逼近确实得到推导，才按相应强度报告�?

**外部起点�?* RF-GPT 已能做频谱图问答，MERLIN 已研究低信噪比下的电磁感知与推理，LEMMO 已能从长 I/Q 中按指令抽取证据并描述跨信号关系。LEMMO 使用 Qwen3.5-4B，可用于同底座比较。分别报告双方原生用法，以及在访问权限、缓存精度、训练预算可匹配时的实验。为双方记录重读、上下文和延迟成本。外部方法未支持的题单列，双方均可评分的题才用于性能比较。[RF-GPT](https://arxiv.org/abs/2602.14833) · [MERLIN](https://em-merlin.github.io/) · [LEMMO](https://arxiv.org/html/2611.09476v1)

**最终盲测�?* 先编码，后揭题。两种解释均与旧波形相容时，只允许再读取一个已录制的候选参考窗。模型先选窗口并写下各解释预期的观测，再揭示真实波形并评分。对照包括随机窗、最近窗以及由物理模型预测能最大程度减少歧义的窗。提前约定主要指标、可接受的误差差距和采集划分。等缓存与观测预算时，比较答案风险、置信度和证据定位；达到约定准确度时，比较所需缓存和额外读取量。天然观测、人工注入和独立采集分别报告。若现有方法打平，保留已验证的边界与反例，继续寻找下一个确有差异的问题�?

完整交付包含可审查的数学结果、带解析真值和独立采集的盲测集、可接入多个已有 AI 的电磁编码器，以及一段支持多轮追问的真实录制演示。[3B�?7B 宿主与模型接入计划](MULTIMODAL_INTEGRATION_PLAN.md)

### 一、我们现在站在哪�?

项目已经走到**可以为电磁模态规定第一组能力，并给出物理判�?*的地方。此前的接收研究留下了相位参考、留出观测、跨帧关系、候选解释和真实采集。这些材料能让我们问一个比“这个码块译对了吗”更直接的问题：同样的输入，哪些物理问题能答，哪些要等另一段证据，编码时扔掉什么会让将来的答案永久丢失�?

|已有研究                 |报告中的发现                                                                                                                                                                                                                           |长期计划继承什�?                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
|---------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|P1–P3 的表示、架构、物理约束与负对照|P1 新种子下 CNN 在两项陌生信道上方向一致地优于同预�?Transformer，校正后均为 p=0.125。P2 的相位等变模型改善了一些合成条件，经典 DFT 仍很强。P3 去掉绝对时延输入并加保留项后，稀疏长时延 BER �?12.291% 降到 6.549%，硬阈�?DFT �?4.919%。还做过稀疏导频、RMS 编码、混合信道、数值精度、单因素失真�?16-QAM 探索。P3-D/E 的标准信道桥接实际完成的是静�?TDL 近似。|复数编码、相位等变读出、匹配先验的经典对照，以及识别位置捷径的方法�?[P1–P3 总览](https://github.com/20171806011/EM-FloatFormer/blob/63ad2ec5b22027ea3d6f15aa50619a9caf6bbcd1/README.md) · [位置捷径](https://github.com/20171806011/EM-FloatFormer/blob/63ad2ec5b22027ea3d6f15aa50619a9caf6bbcd1/P3_A_MECHANISM_REPORT.md) · [TDL 近似的范围](https://github.com/20171806011/EM-FloatFormer/blob/63ad2ec5b22027ea3d6f15aa50619a9caf6bbcd1/exploratory/p3de_tdl_surrogate/P3DE_SURROGATE_REPORT.md) · [16-QAM 探索](https://github.com/20171806011/EM-FloatFormer/blob/63ad2ec5b22027ea3d6f15aa50619a9caf6bbcd1/exploratory/16qam_joint_pa_probe/REPORT.md)                                                                                                                                                                                                             |
|P4–P6 与三�?P6 后续      |P4 的十种子二维模型没有达到预设译码优势。P5/P5-B/P6 逐步对照复数算子、物理估计、奇偶校验和学习模块。P6 在指定模拟中取得收益，局部与注意力对照仍不足以将总体收益归因给注意力。三个后续分别检查了参数化的连续性、条件方差和失效导频的观测模型�?                                                                                                 |把物理输入、编码器和下游译码器的贡献分开。记录缺失样本与测得零值之间的区别�?[P4](https://github.com/20171806011/EM-FloatFormer/blob/63ad2ec5b22027ea3d6f15aa50619a9caf6bbcd1/p4/STAGE3_RESULTS.md) · [P5](https://github.com/20171806011/EM-FloatFormer/blob/63ad2ec5b22027ea3d6f15aa50619a9caf6bbcd1/p5/P5_REPORT.md) · [P5-B](https://github.com/20171806011/EM-FloatFormer/blob/63ad2ec5b22027ea3d6f15aa50619a9caf6bbcd1/p5b/P5B_REPORT.md) · [P6](https://github.com/20171806011/EM-FloatFormer/blob/63ad2ec5b22027ea3d6f15aa50619a9caf6bbcd1/p6/P6_REPORT.md)                                                                                                                                                                                                                                                                                            |
|P7、P8 的三条线           |P7 留出四个导频，先预测再揭示，高分歧比特的概率改变量是低分歧组�?5.52 倍，最终只净少错两个码块。P8 分别考察连续 OFDM 帧和导频位置、从 OFDM 传递状态给单载波，以及离格下的相位传输。跨波形合成实验在稀疏训练条件下，正确历史、清空历史、错配历史的 BLER 分别�?54.84%�?9.47%�?8.71%�?                                                            |已经有“新证据应当改变哪些答案”的实验原型，也有“错误参考会造成什么后果”的对照�?[P7](https://github.com/20171806011/EM-FloatFormer/blob/4dab1ac9ee15b4f1a7f78ae05486c55a2dc3bd72/p7/P7_REPORT.md) · [P8 连续帧](https://github.com/20171806011/EM-FloatFormer/blob/4e7ccdaaacc14cd4101a6f5b824d9974a9255b58/p8/P8_REPORT.md) · [P8 跨波形](https://github.com/20171806011/EM-FloatFormer/blob/ce2ff50e81f6b529ec0311393da0013b0a85b801/p8c/P8C_REPORT.md) · [P8 离格](https://github.com/20171806011/EM-FloatFormer/blob/e76c61eef26151985963460b0ab03684cd562e3e/p8_manifold/P8_MANIFOLD_REPORT.md)                                                                                                                                                                                                                                                                   |
|P9–P11 的跨帧记忆与未知载荷    |P9 表明旧观测是否可靠取决于当前信道。P10 的行内复相关保留路径间的相对相位，同时消去每行公共相位。在规定的合成三径重置实验中，重建分支救回两种旧方法均错�?1,604/2,666 个码块，也使 115 个两者均对的块变错。P11 验证了经过校验的助手行能参与后续估计�?                                                                                       |可以指明某个统计量保存了哪类物理关系，又丢掉哪类信息。评价一次更新时同时记下纠正和新增的错误�?[P9](https://github.com/20171806011/EM-FloatFormer/blob/faeea639727aab72740cdd0a719106f83824226e/p9/P9_REPORT.md) · [P10 复相关实验](https://github.com/20171806011/EM-FloatFormer/blob/9e9d93bcbf6345b8645314b223cd954b2f31e63a/p10/payload_reconstruction/README.md) · [P11](https://github.com/20171806011/EM-FloatFormer/blob/307d62fd89206442960fa0cebdcdc6a0530db366/p11/P11_REPORT.md)                                                                                                                                                                                                                                                                                                                                                                                 |
|P12、两条数学研究与 Stage 5  |P12 修复同一锚点被重复计入等问题。远端有效参考可从内容识别时，全局取证可以严格优于固定局部窗。正确后验融合改善平方风险，严格改善比特判决还取决于判决边界。修复后�?Stage 5 在分布内合成条件下，基线、物理组合、加等变网络�?BLER 分别�?48.94%�?1.58%�?9.54%。分布外的神经增量仍不确定�?                                                                |全局找证据、保留多个解释、一次观测只计一次，以及按问题分别定义损失�?[P12 修复](https://github.com/20171806011/EM-FloatFormer/blob/f566f9b8b0ddfb468a4cb3fd64588bfa46c87212/p12/P12_CONSISTENCY_FIX.md) · [内容寻址](https://github.com/20171806011/EM-FloatFormer/blob/f566f9b8b0ddfb468a4cb3fd64588bfa46c87212/research/content_addressing_scalar/REPORT.md) · [后验融合](https://github.com/20171806011/EM-FloatFormer/blob/f566f9b8b0ddfb468a4cb3fd64588bfa46c87212/research/posterior_fusion/pages/01.md) · [Stage 5](https://github.com/20171806011/EM-FloatFormer/blob/bfe942d94d04004c40575bbb62e57c7e36f3b3e7/research/p12_stage5_fusion_neural/REPORT.md)                                                                                                                                                                                                  |
|四条实测分支、三种数据来�?       |435 MHz 两条分支使用同一�?742 �?16-QAM OFDM 录制�?38 帧和 240 帧是不同划分。Pluto 提供连续时域 QPSK，CTTC 提供多会�?QPSK。另有宽�?Wi-Fi CSI 文件的格式和重复性审计�?                                                                                                          |在不同信号输入层上分别验证可回答的问题；划分时以会话或物理录制为单位。宽带数据盘点供将来选题�?[435 MHz OFDM](https://github.com/20171806011/EM-FloatFormer/blob/ba39dd5968dbaee7b7a226a9bc78d536278fdee8/research/realdata_sdr_ofdm/REPORT.md) · [压力实验](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/REPORT.md) · [Pluto](https://github.com/20171806011/EM-FloatFormer/blob/ade8333fb68126251a83e484b69b821b5d97bcea/research/realdata_pluto_qpsk/REPORT.md) · [CTTC](https://github.com/20171806011/EM-FloatFormer/blob/1887aee6de9649bc0dd8a4f05fe3c4542f3f6d77/research/realdata_cttc_qpsk/REPORT.md) · [宽带样本审计](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/DATASET_DISCOVERY.md)|
|11 �?22 �?A–E 会合     |A/B 核对后验读出并定位同一观测被重复利用的问题。C 的连续频偏精化在部分离格条件有益，在跳变上可能有损。D 给出相位跳变后可辨识量和误码下界。E 给出何时值得切换估计器的理论条件，实�?14 帧合成检验比固定连续方法多错 172 位。会合报告中�?435 MHz 数字是对历史逐帧记录的复算�?                                                                            |优先延伸 D，借用 A/B 的证据核算与 P7 的预测—揭示设计。频偏精化和门控作为具体问题的工具�?[五组会合](https://github.com/20171806011/EM-FloatFormer/blob/e914befb2c03838c5e911e1af55c64230f1f42d6/research/abcde_convergence_2026_11_22/README.md) · [D 组证明](https://github.com/20171806011/EM-FloatFormer/blob/e914befb2c03838c5e911e1af55c64230f1f42d6/research/abcde_convergence_2026_11_22/D/PROOF.md) · [实测输入与复算](https://github.com/20171806011/EM-FloatFormer/blob/e914befb2c03838c5e911e1af55c64230f1f42d6/research/abcde_convergence_2026_11_22/experiments/historical/REAL_INPUTS.md)                                                                                                                                                                                                                                                                      |

**最直接的起点是 D 组的同一段波形�?* 已知非零信道、均匀未编�?16\-QAM、七种等概率非零跳角、无噪无抖动时，已知跳点之后，两个符号的比值仍可精确计算，每个原始标签位的最优平均错误率却是 3/7。若跳点也隐藏，加入一枚保证落在跳变后的已知非零导频，整帧期望 BER �?24\.7347% 降到 2\.0612%。发生跳变的确切时刻仍可能有歧义。这里已经可以对同一段信号连续追问，并逐个说明答案凭什么成立。[五组会合](https://github.com/20171806011/EM-FloatFormer/blob/e914befb2c03838c5e911e1af55c64230f1f42d6/research/abcde_convergence_2026_11_22/README.md) · [D 组证明](https://github.com/20171806011/EM-FloatFormer/blob/e914befb2c03838c5e911e1af55c64230f1f42d6/research/abcde_convergence_2026_11_22/D/PROOF.md) · [实测输入与复算](https://github.com/20171806011/EM-FloatFormer/blob/e914befb2c03838c5e911e1af55c64230f1f42d6/research/abcde_convergence_2026_11_22/experiments/historical/REAL_INPUTS.md)

### 二、我们要争取的贡�?

已有 RF\-GPT 用频谱图进行无线信号问答。MERLIN 研究电磁信号感知与推理及低信噪比条件。LEMMO 已报告从最长八百万个复采样的原�?I/Q 中按指令检索证据，并描述跨信号关系。仅凭“输�?I/Q、接上语言模型、能回答问题”不足以建立本项目的原创性。值得争取的贡献是，针对明确的一族物理问题，给出**有限表示保留了什么、何时必然丢失答案、什么额外观测可以恢复答�?*的定量结果，并在真实波形上检验这些预测。与最接近模型的同题同信息对比将决定可以提出多强的实证主张。[RF\-GPT](https://arxiv.org/abs/2602.14833) · [MERLIN，IEEE/CVF Conference on Computer Vision and Pattern Recognition 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Shen_MERLIN_Building_Low-SNR_Robust_Multimodal_LLMs_for_Electromagnetic_Signals_CVPR_2026_paper.pdf) · [LEMMO](https://arxiv.org/abs/2611.09476)

**用一个共同接口把数学和原型接起来�?* 模型把观测到的复波形 &#40;Y&#41; 编为 &#40;Z=T&#40;Y&#41;&#41;。先问一个问�?&#40;q&#41;，再揭示允许的参�?&#40;A&#41;，向同一�?&#40;Z&#41; 追问。实验时规定能否重读原波形、存储比特数、允许的元数据和各问题的损失。对每个问题分别比较�?&#40;&#40;Z,A&#41;&#41; 与用 &#40;&#40;Y,A&#41;&#41; 所能达到的最优风险。完�?I/Q 是保留信息的参照。在真实采集上最优风险通常未知，改用已知参考、留出的波形和明确的解析估计器评分，把可计算的理论值留给生成模型明确的实验�?

**第一题可以小到手算�?* 两个独立均匀四相符号 &#40;X\_1,X\_2&#41; 经过同一未知均匀四相旋转 &#40;G&#41;，得到无噪观�?&#40;Y\_k=GX\_k&#41;。现在问二者的相对相位，保�?&#40;R=Y\_2\\overline\{Y\_1\}&#41; 的四种状态即可，恰需 2 比特。稍后接收同一相干区间内单位导频的观测 &#40;P=G&#41;，再问第一个符号的原始四相标签。此�?&#40;X\_1=\\overline P Y\_1&#41;。若第一次只保存 &#40;R&#41;，补�?&#40;P&#41; 后仍有四个等可能答案。若同时保存 &#40;&#40;R,Y\_1&#41;&#41;，两题都能答对，恰需 4 比特。后一项也是下界：固定 &#40;P=1&#41; 时，16 种观测对对应 16 种不同的联合答案，任何固定长度、零误差、不能重读波形的编码都须区分它们。本计划另行枚举�?64 种等可能潜在状态来核对这个新例子。它是后续研究的出发点；真正待解的是有噪、部分相干和有限预算下的推广�?

**第一个值得认真攻的推广，是“参考在多远之外还管用”�?* 定义两个时间窗共享同一相位状态的概率或一个可测的相位漂移模型，逐渐拉长窗口间隔，改变参考的信噪比和表示预算。希望推导两条风险曲线：一条对应跨窗相对相位，一条对应补参考后的符号或事件判断。若两种表示在第一条曲线上打平，却在第二条上出现可预测的差距，就得到了对表示的区分。若共享状态的假设在真实采集上失效，测出失效区间本身就能约束后续模型�?

### 三、阶段安�?

|阶段与预计时�?                   |模型将多回答什�?                                    |研究工作、交付与决定                                                                                                                                                            |
|---------------------------|---------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|1 · �?1�? 周，定出可辨识边�?       |“两段信号相对转了多少？给我参考之后，先前那个符号能确定吗？�?             |从四相模型和 D �?16-QAM 推导开始。分别写出只有当前波形、加入同相干区间参考、只给独立参考时的可辨识量。构造两组被某种编码压成同一值、但追加参考后正确答案不同的波形。交付短证明、反例及有噪版本可检验的预测。已有统计充分性和对称性结果在文献中注明�?                                     |
|2 · �?4�? 周，建立小型判别实验       |“哪一段观测支持我的判断？去掉它，哪一个答案会变？�?                  |用共享母波形制作有效参考、独立参考、重复同一参考和无参考四类配对。给解析方法、只有幅度或逐窗归一化的方法、保留复相位的方法完全相同的输入权限。平衡信噪比、窗位、能量和编号。用精确枚举或可控似然验证理论，冻结问题、划分和评分。输出可�?CPU 复跑的小基准�?                                     |
|3 · �?2�? 个月，共享编码器与首次模型接�?   |“同一份表示能答相对关系、预测留出观测，并在参考到来后重新作答吗？�?          |用有限预算的一个复数编码器配几个简单读出，与解析相�?多假设后验、同预算实数网络等对照。冻结表示后测试新参考位置、相位轨迹与未见问题组合，记录每题相对参照的性能差距和保留信息的成本。先�?Ministral 3B，再�?Qwen3.5-4B。冻结编码器与宿主，为每个宿主训练连接层，按相同问题分别测直接读出和语言回答�?                                  |
|4 · �?3�? 个月，在真实录制上兑现一个预�? |“在陌生的一次采集中，哪段已知波形是有效参考？揭示后，先前的预测应怎样修正？�?     |先用已有 Pluto �?CTTC 的已知序列遮住—揭示，A 次采集设参数，B 次采集检验。CTTC 可留整会话；Pluto 可检查连续时域窗�?35 MHz 网格可检验同题的 OFDM 版本。对缺少真值的问题使用留出导频预测或公开序列定义答案。分别汇报自然录制、基于实测的人工干预与独立来源。交付能从原输入重放的至少一项配对外测。|
|5 · �?6�? 个月，从相位扩到事件和关�?   |“这两个窗口里的变化能由同一条相位轨迹解释吗？变化发生在哪一段？另一种解释还剩多少可能？”|先为平滑漂移、离散跳变和两个独立过程构造可区分与观测等价的案例，控制单窗边际特征。随后用同一表示比较两段波形，返回候选解释、支持时间区间与置信度。跨 QPSK/OFDM 只迁移物理定义相同的问题。若现有数据缺少自然事件真值，报告仿真验证与真实留出预测，另找能区分竞争解释的公开采集�?                        |
|6 · �?9�?2 个月，形成可追问的模态原型和论文|“给你一段新录制，请描述有依据的变化，标出证据。再给你另一段，你要修改前面哪句话？�?  |�?3B�?B 原型上扩展多轮事件问答，加入 Qwen3.5-9B �?Qwen3.8-27B，按资源增加跨家族对照。一段录制支持多轮问答，输出具体时间范围、观测依据、可辨识范围和补证后的修订。保留未见录制及未见问题组合做盲评，在共同支持的任务上与公开模型、经典方法和简化版对比。交付理论稿、可复跑的小基准和一个真实录制演示�?                                       |

到第三个月，目标是交�?*一项有假设的信息下界、一套可运行对照和首个接入已有模型的共享编码�?*。到第六个月，目标是让外部读者用真实录制重现一项关于“哪些答案会随证据改变”的预测。后半年建立在该结果或有解释力的反例上。若跨采集的条件差异仍未厘清，时间优先给物理诊断。硬件采集的合适时点，是我们已经说得清缺哪一种观测才能区分两个解释，而已有公开录制恰好没有它�?

**一年以后的方向取决于哪些能力真正迁移�?* 如果同一表示能在独立来源上稳定完成相位与事件问答，再推进到民用宽带被动录制：从“一个信号发生了什么”扩展到“多段成分何时出现、哪些变化同步、哪一段支持这个解释”。先建立可观测条件和真实标签，再检验跨采样率、带宽与波形制式的迁移。此时才值得为多轮问答扩大训练语料和模型容量�?

### 四、真实数据怎样�?

|输入                                                   |可用来做的第一项新检�?                                                      |推断范围                                                                                                                           |
|-----------------------------------------------------|------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
|Pluto 850 MHz 连续 QPSK�?,096 帧，每帧 7,560 个过采样复样点，已做匹配滤波|用源数据中可核对的参考序列，留出一个窗预测其相位；揭示该窗后，判断另一个窗的预测是否改善。检验绝对帧号和固定重复周期能解释多少结果。|能检验时域相干性、重复结构及参考更新；跨设备普适性需要其他采集�?                                                                                              |
|CTTC 三频段、三档名义信噪比、九�?QPSK 会话，含原始 ADC 与官方匹配滤波记�?       |整个会话只作测试。在对齐成功的片段上，比较原始与滤波后视图对同一关系问题的回答，以及哪些细节在处理后消失�?            |两种视图来自同一次采集；以会话和来源汇总。已有研究是会话内划分�?                                                                                              |
|435 MHz 16-QAM OFDM，同步、去循环前缀、傅里叶变换后的复数网格             |先在已有导频及明确可评分的数据位置上遮住—揭示，之后才加受控频偏或相位跳变�?                           |这份数据可检验网格上的相位证据。历�?240 �?B3 对比学习头，在原始工�?BER �?9.1534% �?10.6077%，在注入频偏工况�?26.7733% �?14.4038%�?1 �?22 日会合复算了历史表，未重新运行实�?I/Q 推理。|

不请模型回答没有物理定义的问题。例如“两个窗口是否相同发射机”需要发射机身份真值与恰当观测；本计划先问“是否符合一条共享相位轨迹”，并说明替代模型。对一枚外观完全一样、来源又隐藏的错配导频，模型能有多确定取决于相关性线索与错配先验。已知独立参考和隐藏可靠性的参考要分别建模、分别评分。[435 MHz OFDM](https://github.com/20171806011/EM-FloatFormer/blob/ba39dd5968dbaee7b7a226a9bc78d536278fdee8/research/realdata_sdr_ofdm/REPORT.md) · [压力实验](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/REPORT.md) · [Pluto](https://github.com/20171806011/EM-FloatFormer/blob/ade8333fb68126251a83e484b69b821b5d97bcea/research/realdata_pluto_qpsk/REPORT.md) · [CTTC](https://github.com/20171806011/EM-FloatFormer/blob/1887aee6de9649bc0dd8a4f05fe3c4542f3f6d77/research/realdata_cttc_qpsk/REPORT.md) · [宽带样本审计](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/DATASET_DISCOVERY.md)

### 五、实验怎样表明模型读到了波�?

- **按题评分�?* 相对相位用圆周误差或四相准确率，留出复样本预测用归一化均方误差，候选解释用概率对数损失和校准，跳点定位用时间误差或覆盖区间。另�?BER/BLER 供接收任务复核�?
- **验证依据�?* 对照组拿到相同可见信息。替换或遮住模型声称依赖的窗口，再替换相同长度、相似能量的无关窗，比较答案实际变化。对已有证据的完全重复，精确参考后验应保持一致。若表现超出可辨识上限，优先排查标签、重复载荷和输入元数据�?
- **量化更新�?* 在同一样本上并排记录第一次回答、有效或独立参考带来的改变、哪些答案变对或变错。会话级别给出效应与区间，保留越过相干时间、低信噪比、同步失败和候选解释难分的样本�?
- **控制计算�?* 同时给出完整 I/Q 参照，以及在相同缓存位数、同样允许重读或禁止重读条件下的解析与学习方法。并列参数量、延迟和查询次数。若经典表示能达到理论风险，它就是后续必须保留的对照�?

### 六、第一轮怎样并行

以下是下一轮拟议的五个工作包。起点统一为上面的四相模型�?D 组公开模型。五份结果汇合后再启动正式训练�?

|工作�?       |可以独立完成的部�?                                     |汇合时交什�?                  |
|-----------|-----------------------------------------------|-------------------------|
|1 · 数学     |推导有噪和部分相干时，当前问题与追加参考问题各自的最优风险或�?               |一项有假设的命题、证明与可复算特�?       |
|2 · 反例与现有工作|构造只留幅度、逐窗旋转归一化、错误信道共享假设的反例；核对最接近的无线通信、统计与电磁问答论文|能推翻过强主张的信号对，以及本项目能独立检验的增量|
|3 · 解析对照   |实现四相穷举、保相表示、独立参考与重复证据的精确解；用 P7 的遮住—揭示接口作参照     |同一个微型样本的答案、概率与计算成本       |
|4 · 实测可行�? |查明 Pluto、CTTC 哪些参考序列、输入视图和会话可配对；435 MHz 记录可检哪类题|“问题—输入—真值—划分”表，加一个可复现样例标识|
|5 · 评价     |固定问题句式和输出格式，制定匹配单窗统计的对照、会话留出与证据扰动计分方法          |所有方案都能使用的一份盲测协议草�?       |

第一次汇合围绕同一个波形对完成三件事：数学组列出能排除的解释，基准组给出精确概率，实测组指出同样的观测在真实记录里能否取得。三者有分歧就收窄假设。这一步可以把一个有趣的信号处理现象变成第一条可供独立检验的科学主张�?

**防止研究兜圈子的约定�?* 每轮开始写一句“做完以后，面对同一段波形，模型能多回答哪个问题”；每轮结束列出一个成功样例、一个失败样例和会随之改变的研究决定。若连续两轮都只能报告同一�?BER 指标的小幅变化，就把该接收优化归档，回到尚未回答的问题。若理论给出不可辨识结论，保留这个结论和它所要求的最小新增观测，这也是主线成果�?

### 附录 · GitHub 分支快照

下列链接固定到审阅时各分支的末端提交。最新可见提交为 2026 �?11 �?22 �?19:47（伦敦时间）的五组会合。主分支末端�?11 �?18 日，单看仓库首页会漏掉后续阶段�?

|分支                                              |研究内容                    |固定版本的报�?                                                                                                                                                  |
|------------------------------------------------|------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
|`main`                                          |P1–P6 合成接收研究，含早期补充和负结果  |[63ad2ec5](https://github.com/20171806011/EM-FloatFormer/blob/63ad2ec5b22027ea3d6f15aa50619a9caf6bbcd1/README.md)                                         |
|`research/p6-math-followup-2026-11-19`          |平滑符号选择与波形化简             |[c699680d](https://github.com/20171806011/EM-FloatFormer/blob/c699680dd84560747ddf40ecacd59821987bf545/p6_followup/REPORT.md)                             |
|`research/p6-soft-information-2026-11-19`       |符号条件方差与相位精�?            |[bb9396f0](https://github.com/20171806011/EM-FloatFormer/blob/bb9396f08132bb5f586ff728dcfa4e7872201c7b/p6_softinfo/REPORT.md)                             |
|`research/p6-observation-consistency-2026-11-19`|失效导频与观测一致�?             |[a4e66848](https://github.com/20171806011/EM-FloatFormer/blob/a4e6684830f68e6dfe715470dafb4ff95972e508/p6_consistency/REPORT.md)                          |
|`research/p7-multihypothesis-routing`           |多假设与留出导频更新              |[4dab1ac9](https://github.com/20171806011/EM-FloatFormer/blob/4dab1ac9ee15b4f1a7f78ae05486c55a2dc3bd72/p7/P7_REPORT.md)                                   |
|`research/p8-continuous-ofdm-tracking`          |连续 OFDM 帧间状态与导频布局       |[4e7ccdaa](https://github.com/20171806011/EM-FloatFormer/blob/4e7ccdaaacc14cd4101a6f5b824d9974a9255b58/p8/P8_REPORT.md)                                   |
|`research/p8c-ofdm-single-carrier`              |OFDM 到单载波的传播状态传�?       |[ce2ff50e](https://github.com/20171806011/EM-FloatFormer/blob/ce2ff50e81f6b529ec0311393da0013b0a85b801/p8c/P8C_REPORT.md)                                 |
|`research/p8-coherent-transport-manifold`       |离格相位传输与连续揭示证�?          |[e76c61ee](https://github.com/20171806011/EM-FloatFormer/blob/e76c61eef26151985963460b0ab03684cd562e3e/p8_manifold/P8_MANIFOLD_REPORT.md)                 |
|`research/p9-cross-waveform-trust-gating`       |旧信道是否仍可信                |[faeea639](https://github.com/20171806011/EM-FloatFormer/blob/faeea639727aab72740cdd0a719106f83824226e/p9/P9_REPORT.md)                                   |
|`research/p10-payload-channel-reconstruction`   |未知载荷辅助信道重建              |[9e9d93bc](https://github.com/20171806011/EM-FloatFormer/blob/9e9d93bcbf6345b8645314b223cd954b2f31e63a/p10/README.md)                                     |
|`research/p11-wavefront-payload-assimilation`   |经过校验的助手行参与估计            |[307d62fd](https://github.com/20171806011/EM-FloatFormer/blob/307d62fd89206442960fa0cebdcdc6a0530db366/p11/P11_REPORT.md)                                 |
|`research/p12-first-principles-diagnostics`     |离格接收修复、内容寻址与后验融合的数学结果   |[f566f9b8](https://github.com/20171806011/EM-FloatFormer/blob/f566f9b8b0ddfb468a4cb3fd64588bfa46c87212/research/content_addressing_scalar/REPORT.md)      |
|`research/p12-stage5-offgrid-neural`            |P12 修复后的物理与可学习模型对照      |[bfe942d9](https://github.com/20171806011/EM-FloatFormer/blob/bfe942d94d04004c40575bbb62e57c7e36f3b3e7/research/p12_stage5_fusion_neural/REPORT.md)       |
|`research/realdata-sdr-ofdm`                    |435 MHz OFDM 接收实验与输入校�? |[ba39dd59](https://github.com/20171806011/EM-FloatFormer/blob/ba39dd5968dbaee7b7a226a9bc78d536278fdee8/research/realdata_sdr_ofdm/REPORT.md)              |
|`research/realdata-ofdm-theory-and-stress`      |435 MHz 同一录制的压力研究与宽带数据盘点|[5f88a5be](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/REPORT.md)|
|`research/realdata-pluto-qpsk`                  |Pluto 连续 QPSK           |[ade8333f](https://github.com/20171806011/EM-FloatFormer/blob/ade8333fb68126251a83e484b69b821b5d97bcea/research/realdata_pluto_qpsk/REPORT.md)            |
|`research/realdata-cttc-indoor-qpsk`            |CTTC 三频段九会话 QPSK        |[1887aee6](https://github.com/20171806011/EM-FloatFormer/blob/1887aee6de9649bc0dd8a4f05fe3c4542f3f6d77/research/realdata_cttc_qpsk/REPORT.md)             |
|`research/abcde-convergence-2026-11-22`         |五组数学会合与配对实�?            |[e914befb](https://github.com/20171806011/EM-FloatFormer/blob/e914befb2c03838c5e911e1af55c64230f1f42d6/research/abcde_convergence_2026_11_22/README.md)   |
{
  "experiment": "EM-FloatFormer Stage 3D (Qwen3.5-4B signed phase vs zero arm)",
  "model_id": "Qwen/Qwen3.5-4B",
  "hf_sha": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
  "encoder_name": "equivariant_b8_s17",
  "encoder_sha256": "e67d9fbafb90791ab19549de7789798d818286d7f52d7ac566cc39df71e78356",
  "chat_template_sha256": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
  "thinking_mode": "disabled (enable_thinking=False -> <think>\\n\\n</think>\\n\\n)",
  "host_sha256": "f39c9e1b7a5719c1397dcdce42d221215fb20edadb881d2f81e745673636f64d",
  "versions": {
    "transformers": "5.24.1",
    "torch": "2.15.0+cu132",
    "numpy": "2.3.5",
    "scipy": "1.17.0",
    "gpu_name": "NVIDIA A100-SXM4-80GB"
  },
  "locked_hyperparameters": {
    "arm_names": [
      "phase",
      "zero"
    ],
    "connector_seeds": [
      202611,
      202622,
      202633
    ],
    "trainable_params": 2184964,
    "epochs": 8,
    "batch_size": 32,
    "micro_batch_size": 16,
    "lr": 0.0015,
    "max_steps_per_epoch": 160,
    "train_scorer": "kv_cached",
    "eval_scorer": "kv_cached",
    "eval_every_epochs": 2
  },
  "checkpoints": {
    "phase_s202611": {
      "filename": "connector_phase_s202611.pt",
      "checkpoint_sha256": "088d4ad34e1152ff000959a37bf784a8f45e326bdd2e483fdf755f82405f7866",
      "best_epoch": 7,
      "best_sel_score": 1.3809686587691041,
      "temp_q3": 0.5098932035670315,
      "temp_q2": 3.9999830531425227,
      "dev_q3_cal_nll": 1.375347147854539,
      "dev_q2_cal_nll": 1.3866422230228832,
      "host_sha256_after": "f39c9e1b7a5719c1397dcdce42d221215fb20edadb881d2f81e745673636f64d"
    },
    "phase_s202622": {
      "filename": "connector_phase_s202622.pt",
      "checkpoint_sha256": "ba275787201abf5f1837bd9ebcc959dab24746f9ffa1a08a5df93b0e8a263ae0",
      "best_epoch": 8,
      "best_sel_score": 1.338227526868415,
      "temp_q3": 0.8489407339257738,
      "temp_q2": 1.1069058926498243,
      "dev_q3_cal_nll": 1.285269302113726,
      "dev_q2_cal_nll": 1.3565327978349748,
      "host_sha256_after": "f39c9e1b7a5719c1397dcdce42d221215fb20edadb881d2f81e745673636f64d"
    },
    "phase_s202633": {
      "filename": "connector_phase_s202633.pt",
      "checkpoint_sha256": "75f0022894bde60b6e69673d670604b4f0a8cca7847d8df22339e36ba3ddd92f",
      "best_epoch": 8,
      "best_sel_score": 1.345658880106861,
      "temp_q3": 0.6574859852227614,
      "temp_q2": 1.0038161968358967,
      "dev_q3_cal_nll": 1.3001861761732214,
      "dev_q2_cal_nll": 1.3714957209402086,
      "host_sha256_after": "f39c9e1b7a5719c1397dcdce42d221215fb20edadb881d2f81e745673636f64d"
    },
    "zero_s202611": {
      "filename": "connector_zero_s202611.pt",
      "checkpoint_sha256": "3d09eded33b58b928191e54d344381ae141b70473d0409cd660d259ec742b77c",
      "best_epoch": 8,
      "best_sel_score": 1.360639128458629,
      "temp_q3": 1.0413709772961914,
      "temp_q2": 1.2209637372368802,
      "dev_q3_cal_nll": 1.3349880521106885,
      "dev_q2_cal_nll": 1.3793762964672136,
      "host_sha256_after": "f39c9e1b7a5719c1397dcdce42d221215fb20edadb881d2f81e745673636f64d"
    },
    "zero_s202622": {
      "filename": "connector_zero_s202622.pt",
      "checkpoint_sha256": "a8eb55bfad0d266cbf7e9dc9712b2d32b162374a39ab2898a7f54fa8724c3e98",
      "best_epoch": 8,
      "best_sel_score": 1.3124549620599506,
      "temp_q3": 0.7826950522516535,
      "temp_q2": 1.4118146530519748,
      "dev_q3_cal_nll": 1.233350441464049,
      "dev_q2_cal_nll": 1.3069782597576265,
      "host_sha256_after": "f39c9e1b7a5719c1397dcdce42d221215fb20edadb881d2f81e745673636f64d"
    },
    "zero_s202633": {
      "filename": "connector_zero_s202633.pt",
      "checkpoint_sha256": "ea7aebcc6d53bc7161f07ed38c0d5b21e473ff8b56aa3b5af1db61f80f9eb4b3",
      "best_epoch": 8,
      "best_sel_score": 1.341565335646975,
      "temp_q3": 0.6733062486614652,
      "temp_q2": 1.2734733247377854,
      "dev_q3_cal_nll": 1.292657943214399,
      "dev_q2_cal_nll": 1.3811923802166672,
      "host_sha256_after": "f39c9e1b7a5719c1397dcdce42d221215fb20edadb881d2f81e745673636f64d"
    }
  },
  "dev_pilot": {
    "namespace": 940202,
    "blocks": 16,
    "mothers": 1024,
    "pilot_block_seeds": [
      2338220280,
      1372146723,
      1251899241,
      1019832785,
      2632972314,
      1698518890,
      2353612898,
      1409965763,
      3844068644,
      2792309556,
      1248517795,
      579468612,
      575362531,
      3621001698,
      787505858,
      3687402811
    ],
    "observed_acc_diff_mean": -0.03352864583333334,
    "observed_acc_diff_block_sd": 0.014638601593173158,
    "observed_nll_diff_mean": -0.02437774790791658,
    "observed_nll_diff_block_sd": 0.013414584590285893,
    "chi2_upper_multiplier_df15_alpha05": 1.437305501981677,
    "conservative_upper_sd_hit": 0.02104014261118552,
    "conservative_upper_sd_nll": 0.019280856238416536,
    "work_e_scenario": {
      "q": 0.35,
      "rho": 0.01,
      "nll_mother_sd": 0.5,
      "pilot_block_sd": [
        0.02104014,
        0.01928086
      ],
      "sd": {
        "accuracy": 0.09441431300390846,
        "nll": 0.07979465834252315
      },
      "normal_by_endpoint": {
        "accuracy": 129,
        "nll": 52
      },
      "t_by_endpoint": {
        "accuracy": 131,
        "nll": 54
      },
      "normal_blocks": 129,
      "normal_mothers": 8256,
      "t_blocks": 131,
      "t_mothers": 8384,
      "normal_at_normal": {
        "accuracy": {
          "power": 0.9504225431012328,
          "plugin_half_width": 0.016292625434745873,
          "expected_half_width": 0.016292625434745873
        },
        "nll": {
          "power": 0.999905603276502,
          "plugin_half_width": 0.013769781706873525,
          "expected_half_width": 0.013769781706873525
        }
      },
      "t_at_normal": {
        "accuracy": {
          "power": 0.9475722738758099,
          "plugin_half_width": 0.016448130307161433,
          "expected_half_width": 0.016416036730692727
        },
        "nll": {
          "power": 0.9998880532300913,
          "plugin_half_width": 0.013901207311426547,
          "expected_half_width": 0.013874083288724564
        }
      },
      "t_at_selected": {
        "accuracy": {
          "power": 0.9505118697757498,
          "plugin_half_width": 0.016319693256397413,
          "expected_half_width": 0.016288339697408512
        },
        "nll": {
          "power": 0.9999060352925451,
          "plugin_half_width": 0.013792658191508557,
          "expected_half_width": 0.013766159597728204
        }
      },
      "t_at_previous": {
        "accuracy": {
          "power": 0.9490618189573753,
          "plugin_half_width": 0.01638353419239668,
          "expected_half_width": 0.016351814221143575
        },
        "nll": {
          "power": 0.9998974312902781,
          "plugin_half_width": 0.013846613630194187,
          "expected_half_width": 0.01381980536152976
        }
      }
    },
    "b_locked": 131
  },
  "b_locked": 131,
  "blind_seed_manifest_sha256": "b6c075162010a0d6e72bb07cd32a4a01aed24e1abca8be2f949846d258c0f7e7",
  "dual_practical_threshold": {
    "acc_gain_t95_lo_gt": 0.02,
    "nll_improv_t95_lo_gt": 0.02,
    "logical_pass": "both"
  }
}
# EM-FloatFormer Stage 3D 实验报告：冻�?Qwen3.5-4B �?s17 缓存下的有符号频偏显式相位向量盲�?

> **研究分支**：`research/stage3d-qwen-signed-phase`（基�?Stage 3C 整合提交 [`4fe9162b`](https://github.com/20171806011/EM-FloatFormer/tree/4fe9162bdbf93fb0059cc53aec264153dbb67d0d) 建立，未修改任何既往阶段代码与数据）  
> **盲测前锁定提交（Commit #1�?*：`a1fac7a8cfe5c2f6011bdae0080e8df2e4e68215`（全�?6 个连接层 checkpoint SHA-256、温度参数、样本量 `B_locked = 131` 与盲测区组种子清单在揭盲前写�?Git 历史�?

---

## 首页执行摘要与双门槛判定

本轮实验回答单一主问题：**在完全冻�?`Qwen/Qwen3.5-4B` �?`equivariant_b8_s17` 离散波形编码器、两臂均从中文题面获得完全相同的有符号频偏元数据且训练数据与更新预算逐项相同的条件下，仅将连接层�?6 个公开数值槽由全零（`zero` 对照臂）替换为对应时�?$t \in \{h, h+\ell, t_p\}$ 的显式相位向�?$[\cos(\omega t), \sin(\omega t)]$（`phase` 相位臂），能否提高对未见频偏�?Q3-repeat 相位扇区预测�?*

### 主检验格 `G1` 汇总（`|ω| = 0.40`, `h = 2`, `�?= 2`, `t_p = 4.0`, `|ω|(h+�? = 1.60 rad`, `σ²t = 0.04`�?

| 评估对象 / 对比�?| 准确�?(`Acc`) | 负对数似�?(`NLL`) | `Brier` | 10-bin `ECE` | 说明 / 95% 区间 |
|---|---:|---:|---:|---:|---|
| **相位�?`phase`（三套种子等权汇总）** | **23.80%** (`0.23799`) | **1.4452** (`1.44523`) | `0.7845` | `0.1186` | `131` 独立区组 × `64` 母波�?= `8,384` 条（`150,912` 逻辑行）；`-ω: 24.95% / 1.4547`，`+ω: 22.65% / 1.4358` |
| **对照�?`zero`（三套种子等权汇总）** | **19.41%** (`0.19406`) | **1.4809** (`1.48088`) | `0.8010` | `0.1827` | `131` 独立区组 × `64` 母波�?= `8,384` 条（`150,912` 逻辑行）；`-ω: 18.80% / 1.4678`，`+ω: 20.02% / 1.4939` |
| **配对差值：准确率增�?`phase �?zero`** | **+4.39 pp** (`+0.04393`) | �?| �?| �?| **95% 配对 `t` 区间**：`[+4.11, +4.68] pp`（区�?SD `0.01644`，SE `0.00144`）；**区组 Bootstrap**：`[+4.11, +4.67] pp` |
| **配对差值：NLL 改善 `zero �?phase`** | �?| **+0.0357 nats** (`+0.03565`) | �?| �?| **95% 配对 `t` 区间**：`[+0.0330, +0.0383] nats`（区�?SD `0.01556`，SE `0.00136`）；**区组 Bootstrap**：`[+0.0330, +0.0383] nats` |
| 同期同母波形 `s17` CPU 物理读出 | **64.62%** (`0.64617`) | **0.9411** (`0.94113`) | `0.5098` | `0.0009` | 真实连接层相�?CPU 物理读出差距：`phase �?0.82 pp / -0.5041 nats`；`zero �?5.21 pp / -0.5398 nats` |
| 同期同母波形 `full_iq` 已知参数全波形解 | **64.64%** (`0.64641`) | **0.9290** (`0.92904`) | `0.5032` | `0.0046` | 物理上限基线 |
| 四选一均匀随机基线 | `25.00%` | `1.3863` ($\ln 4$) | `0.7500` | `0.0000` | 无信息猜测参考线 |

### 三个预登记连接层种子的各自估计与离散程度（`G1`�?

| 种子 (`seed`) | `phase` Acc / NLL (`ECE`) | `zero` Acc / NLL (`ECE`) | 准确率增�?`phase �?zero` (`[95% t-CI]`) | NLL 改善 `zero �?phase` (`[95% t-CI]`) |
|---|---:|---:|---:|---:|
| `202611` | `26.59% / 1.3950` (`0.0625`) | `19.98% / 1.4436` (`0.1274`) | `+6.62 pp` (`[+6.12, +7.11] pp`) | `+0.0486 nats` (`[+0.0457, +0.0516] nats`) |
| `202622` | `21.23% / 1.4814` (`0.1937`) | `18.17% / 1.5291` (`0.2486`) | `+3.06 pp` (`[+2.68, +3.44] pp`) | `+0.0477 nats` (`[+0.0432, +0.0521] nats`) |
| `202633` | `23.57% / 1.4593` (`0.1219`) | `20.07% / 1.4699` (`0.1722`) | `+3.50 pp` (`[+3.06, +3.95] pp`) | `+0.0106 nats` (`[+0.0068, +0.0145] nats`) |
| **跨种子离散度** | �?| �?| **SD `1.94 pp`**，范�?`[+3.06, +6.62] pp` | **SD `0.0217 nats`**，范�?`[+0.0106, +0.0486] nats` |

> [!IMPORTANT]
> **预设实用双门槛判定：`PASS BOTH`（形式通过，但须结合下方机制澄清理解）**
> 1. **形式判定**：`acc_t95_lo = +4.11 pp > +2.00 pp`（通过）且 `nll_t95_lo = +0.0330 nats > +0.0200 nats`（通过）；三套种子在准确率�?NLL 上全部呈同向正改善�?
> 2. **关键科学澄清（不可将形式显著误读为真正的外推物理建模能力�?*�?
>    - 在主�?`G1` 上，**两臂的绝对准确率�?NLL 均劣于四选一随机猜测（`25.00% / 1.3863`�?*，仅达同缓存 CPU 物理读出（`64.62%`）的约三分之一�?
>    - 结合 2×2 探索格（�?3 节）�?Work F 机制探针（第 4 节）可见：在训练累计转角 $|ω|(h+\ell) \le 1.25\text{ rad} < \pi/2$ 的支持域内（`G2` �?`G4`），**`zero` 对照臂反而显著高�?`phase` 臂（`G2`: `36.36%` vs `33.88%`�?*；而一旦进入累计转�?`1.60 rad > π/2` 的强外推区（`G1` �?`G3`），未学会“跨 $90^\circ$ 象限折返”的两臂均跌破随机线，其中在支持域拟合更锐利�?`zero` 臂因反相关预测跌得更深（`19.41%` / `19.74%`），�?`phase` 臂因仅学会了微弱的物理方向响应（`gain = +0.0098`）加上略微保守的局部预测，少跌了约 3�? 个百分点�?
> 3. **核心机制两问的直接回�?*�?
>    - **宿主概率是否朝物理上正确的方向移动？** **是，但效应极弱�?* �?`±0.40` �?`phase` 对匹配物理对比度的方向增益仅 `+0.0098`（约为理想物理参�?`1.0` �?1%），交换告知符号�?NLL 劣化 `+0.0184 nats`；虽明显强于 `zero` 臂（增益 `+0.0050`、交换劣�?`+0.0074 nats`），且单模态矛盾探针证明宿主对 6 个连续数值相位槽的敏感度（翻转相位槽引起 `TV = 0.0371`）高于对中文频偏文本数字的敏感度（翻转文本引�?`TV = 0.0281`），但两者均未形成有效的 $\pi/2$ 跨象限三角旋转算子�?
>    - **宿主是否辨认出迟到导�?Q2 的新证据�?* **尚未成功辨认第二次独立采样的新增信息�?* 在未�?`C4`（`ω = +0.35, t_p = 4`）条件上，物理核从单次有效导频到两次独立重采样的准确率显著跃�?`+7.81 pp`（`66.41% �?74.22%`, `NLL 0.8192 �?0.6632`）；�?Qwen 相位臂在加入 `valid` 导频后虽�?`none` 略有上升（`24.80% �?28.78%`, `+3.97 pp`，但 NLL 微升 `-0.0040`），再加入第二条独立有效采样 `fresh_resample` 后准确率反而回�?**`-2.34 pp`**（`28.78% �?26.43%`, `NLL 1.3889 �?1.4056`）；`zero` 臂亦表现�?`valid 26.63% �?fresh_resample 25.78%`（`-0.85 pp`）�?

---

## 1. 宿主与编码器锁定、Token 核对、显存故障排查与算力账单

### 1.1 冻结组件、词表与 Chat Template 审计（[smoke_audit.json](results/smoke_audit.json)、[PROTOCOL_LOCK.json](PROTOCOL_LOCK.json)�?

| 审计项目 | 实测锁定�?|
|---|---|
| **语言宿主模型** | `Qwen/Qwen3.5-4B`（HF pinned revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`，已验证丢弃未用视觉编码�?`lm.model.visual = None` 并保�?`tie_word_embeddings`�?|
| **宿主参数与指�?* | 冻结参数 `4,205,751,296`；训练前后采样指�?SHA-256 严格一致：`f39c9e1b7a5719c1397dcdce42d221215fb20edadb881d2f81e745673636f64d` |
| **波形编码�?* | `equivariant_b8_s17`�?6 个离散�?/ 每窗 4 bit，文�?SHA-256：`e67d9fbafb90791ab19549de7789798d818286d7f52d7ac566cc39df71e78356`�?|
| **连接层可训练参数** | **`2,184,964`**（占宿主�?`0.052%`；含新增 6�?28 `phase_mlp` �?`phase_scale`�?*`phase` �?`zero` 两臂结构与参数量 100% 相同**�?|
| **Native Chat Template & Thinking** | Template SHA-256 `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715`；显式关�?thinking（`enable_thinking=False`），严格�?`<|im_start|>assistant\n<think>\n\n</think>\n\n` 结尾（`ends_with_disabled_think = true`�?|
| **软件运行环境** | Modal `NVIDIA A100-SXM4-80GB` (`81,179.8 MB`), `torch 2.15.0+cu132`, `transformers 5.24.1`, `flash-linear-attention 0.6.0`, `numpy 2.3.5`, `scipy 1.17.0`, `torch.bfloat16` |
| **模型加载耗时与常驻显�?* | 加载 `57.69 s`，常驻权重占显存 `8,021.9 MB` |
| **Smoke 前向/反向/重载审计** | 冻结宿主非零梯度�?`host_grads_nonzero = 0`；连接层 29/29 个参数张量全非零有限梯度；断点保存重载最大概率差 `reload_max_prob_diff = 0.0`；Q3 贪心生成样本：`扇区 2�?135°�?-135°）` |

#### 12 个中文候选串逐一 Token 审计（均非空、边界零合并�?

| 任务 | `physical_index` | 中文完整候选文�?| Token 序列（长度） | 边界精确拼接 |
|---|---:|---|---|---:|
| `Q1` | `0 / 1 / 2 / 3` | `0圈（0°）` / `1圈（+90°）` / `2圈（180°）` / `3圈（+270°）` | 分别长度 `6, 8, 8, 9` | 4/4 `True` |
| `Q2` | `0 / 1 / 2 / 3` | `符号0�?）` / `符号1（i）` / `符号2�?1）` / `符号3�?i）` | 分别长度 `5, 5, 6, 6` | 4/4 `True` |
| `Q3` | `0 / 1 / 2 / 3` | `扇区0�?45°�?45°）` / `扇区1�?45°�?135°）` / `扇区2�?135°�?135°）` / `扇区3�?135°�?45°）` | 分别长度 `14, 15, 16, 15` | 4/4 `True` |

### 1.2 工程关键排查：Qwen3.5 混合线性注意力引用环与 BF16 批并行数值一致�?

1. **参照引用环导致的 `71.64 GiB` OOM 定位与根治（[qwen_connector.py](qwen_connector.py)、[bench_steady.json](results/bench_steady.json)�?*�?
   - 初版 `_make_differentiable_qwen35_cache()` 将闭�?`_update_conv(..., _l=layer)` 绑定至每�?`LinearAttentionLayer` 实例，在 Python 中制造了 `layer -> update_conv_state -> _l -> layer` 直接引用环；�?`cand_out` �?`use_cache=False` 仍触�?`cache.update_recurrent_state` 写入计算图张量后，连续微批的激活图无法被引用计数即时回收，最终在首轮 Seed 1 尝试中飙升至 `71.64 GiB` OOM�?
   - **修复方案**：改用模块级缓存子类 `_OutOfPlaceLinearAttentionLayer(LinearAttentionLayer)` 切换 `layer.__class__`（零实例闭包引用环），同时使用原�?`F.log_softmax(..., dim=-1, dtype=torch.float32)` 避免全词�?`.float()` 复制，并�?`cand_out` 前及�?`del pref_out, out_cache, pref_cache`�?
   - **实测验证**：在最长输入组 `Q3_after_pilot`（`351` 字符 + `1` 条参考）�?`Q2_fresh_resample`（`2` 条参考）上连�?4 个微批，分配显存严格恒定�?**`[8055.2, 8055.2, 8055.2, 8055.2] MB`**（零泄漏）；`micro_batch_size=16` 训练峰值稳定在 **`52,972.5 MB`**（稳�?`1,975.7 ms / 32-ex step`），盲测推理 `b=64` 峰�?**`45,409.9 MB`**（`23.88 ms/例`）�?
2. **`bfloat16` + FLA 核的批大小一致与跨批差异透明报告（[direct_vs_table_parity.json](results/direct_vs_table_parity.json)�?*�?
   - 在与 256 查表完全相同�?**`matched_batch_size = 64`** 条件下，直接前向�?256 查表�?G1 盲测 256 条抽样核对中取得 **`max_abs_ll_diff = 0.0`**�?*`max_abs_prob_diff = 0.0`**�?*`argmax_agreement = 1.0`**�? �?`(arm, ±0.40)` 全部精确为零）�?
   - 同时记录了跨批大小的 BF16/Triton 矩阵分块舍入噪声：在训练前随机权重上（`bench_steady.json`），`b=1/16` �?`b=64` �?`max_abs_ll_diff` �?`0.4816�?.4830`（`argmax_agreement = 1.0`）；但在训练后预测高度接近均匀的盲测模型上（`direct_vs_table_parity.json` �?`cross_batch_b16_vs_b64`），`b=16` vs `b=64` �?`max_abs_ll_diff` 降至 `0.115�?.289`（`max_abs_prob_diff = 0.024�?.067`），但因近四选一均分边界细微扰动，导�?`argmax_agreement` �?`phase -0.40` �?`0.6875`（`phase +0.40: 0.8750`, `zero -0.40: 1.0000`, `zero +0.40: 0.9375`）。全盲测统一度量（表生成与全盲测前向）均统一运行在固�?`batch_size = 64` 下，确保内部严格自洽�?

### 1.3 Modal 免费额度算力账单（全程零付费�?

| 费用项目 | Modal 计费账单记录 | 说明 |
|---|---:|---|
| 临时计算作业 (`Ephemeral Apps`) | **`$17.12`** | 覆盖 Smoke、Bench（含早先 1 �?OOM）�? 次完整训练、Dev Pilot、全量盲测以�?1 次提前中止的历史 Ministral 检�?|
| 持久化卷存储 (`Volumes`: `qwen3-model-cache` + `stage3d-results`) | **`$1.34`** | Free Storage 全额抵扣 |
| 网络流出 (`Network Egress`) | **`$0.01`** | Network Egress Allowance 全额抵扣 |
| **实际账单应付 (`Billed Cost`)** | **`$0.00`** | **完全�?`$30.00` Modal 免费计算额度内完成（尚余�?`$12.88` 免费计算额度�?* |

---

## 2. 两臂对照设置�? 次训练过程与样本量盲前锁�?

### 2.1 六个数值槽设计与零泄露契约

- 以第一窗为时间零点 $t_1 = 0$，分别取 $t \in \{h, h+\ell, t_p\}$ 计算 6 维实数值槽�?
  - **`phase` 相位�?*�?[\cos(\omega h), \sin(\omega h), \cos(\omega(h+\ell)), \sin(\omega(h+\ell)), \cos(\omega t_p), \sin(\omega t_p)]$�?
  - **`zero` 对照�?*�?[0, 0, 0, 0, 0, 0]$�?
  - **严格共享控制**：除�?6 个槽外，相同�?`phase_mlp` 结构、相同的 `(z1, z2)` 编码缓存、相同的中文题面与有符号 $\omega$（例�?`omega=+0.40rad/步` �?`-0.40rad/步`）、相同的硬物理扇区标签、相同的训练样本顺序与优化器预算，彻底排除真实布朗路径、未揭示导频与工�?ID 标签�?
- **训练支持�?*：延�?Stage 3 �?`C1 / C2 / C3` 回放（每条件 512 母波形）并按 Work F 加入有符�?CFO 监督�?|\omega| \in \{0, 0.10, 0.25\}$，时延组�?$(h, \ell, t_p) \in \{(2, 2, 4), (2, 3, 1)\}$（训练最大累计转�?$0.25 \times 5 = 1.25\text{ rad}$），每条�?512 母波形（开发集对应 $|\omega| = 0.18$ �?`C1/C2/C3`，每条件 128 母波形）�?
- **锁定超参数（六次训练逐项一致）**：`epochs = 8`, `max_steps_per_epoch = 160`（共 `1,280` �?= `40,960` 样本/次）, `batch_size = 32`, `micro_batch_size = 16`, `lr = 1.5e-3`（AdamW + 5% warmup + cosine 衰减），`train_scorer = eval_scorer = kv_cached`, `eval_every_epochs = 2`�?

### 2.2 6 次连接层训练收敛与开发集标量温度锁定（[training_summary.json](results/training_summary.json)�?

| �?/ 种子 | 最�?Epoch | 开发筛选分 (`sel_score`) | 校准�?`dev_cfo_q3_nll` | 校准�?`dev_q3_cal_nll` (`T_q3`) | 校准�?`dev_q2_cal_nll` (`T_q2`) | 耗时 (s) / 峰�?VRAM | Checkpoint SHA-256（前 16 位） |
|---|---:|---:|---:|---:|---:|---:|---|
| `phase / 202611` | `7` | `1.38097` | `1.37762` (`29.62%`) | `1.37535` (`0.5099`) | `1.38664` (`4.0000`) | `2884.0 s` / `52,972.5 MB` | `088d4ad34e1152ff` |
| `phase / 202622` | `8` | `1.33823` | `1.31390` (`35.70%`) | `1.28527` (`0.8489`) | `1.35653` (`1.1069`) | `3026.2 s` / `52,972.5 MB` | `ba275787201abf5f` |
| `phase / 202633` | `8` | `1.34566` | `1.32520` (`35.00%`) | `1.30019` (`0.6575`) | `1.37150` (`1.0038`) | `3217.8 s` / `52,972.5 MB` | `75f0022894bde60b` |
| `zero / 202611` | `8` | `1.36064` | `1.34627` (`35.68%`) | `1.33499` (`1.0414`) | `1.37938` (`1.2210`) | `2908.7 s` / `52,972.5 MB` | `3d09eded33b58b92` |
| `zero / 202622` | `8` | `1.31245` | `1.26930` (`39.10%`) | `1.23335` (`0.7827`) | `1.30698` (`1.4118`) | `3059.1 s` / `52,972.5 MB` | `a8eb55bfad0d266c` |
| `zero / 202633` | `8` | `1.34157` | `1.31870` (`32.40%`) | `1.29266` (`0.6733`) | `1.38119` (`1.2735`) | `2983.0 s` / `52,972.5 MB` | `ea7aebcc6d53bc71` |

*注：全部 6 次训练的 `host_sha256_after` 与初始完全一致；仅在训练预算结束后的独立开发集 `dv_groups` 全量拟合标量温度 `(T_q3, T_q2)`，训练期损失均在 `1.5�?.6` 顺利降至 `1.29�?.38`�?

### 2.3 揭盲前独立开发先导方差检查与 `B_locked` 锁定（[dev_pilot_lock.json](results/dev_pilot_lock.json)�?

- 在独立先导命名空�?`NS_DEV_PILOT = 940202`（`16` 个独立区�?= `1,024` 条母波形，`|ω| = 0.18, h = �?= 2, t_p = 4`）上运行已锁定的 6 个模型：
  - 实测先导区组均值差：`acc_diff_mean = -3.35 pp`（`-0.03353`），`block_sd = 0.01464`；`nll_diff_mean = -0.02438`，`block_sd = 0.01341`�?
  - �?`df = 15, α = 0.05` �?$\chi^2$ 单侧 95% 保守上界乘子 `1.43731`：保守上界标准差 `sd_acc_upper = 0.02104`、`sd_nll_upper = 0.01928`�?
  - 调用 Stage 3C `research/stage3c/work_results/E/verify.py --pilot-block-sd 0.02104014 0.01928086`，得 `t_blocks = 131`（效能：准确�?`0.9505`、NLL `0.9999`，半�?`±0.01632` / `±0.01379`），由此在揭盲前锁定 **`B_locked = max(131, 131) = 131` 个区组（`8,384` 条母波形�?*，并生成 `blind_seed_manifest.json`（SHA-256：`b6c075162010a0d6e72bb07cd32a4a01aed24e1abca8be2f949846d258c0f7e7`），连同完整代码�?`PROTOCOL_LOCK.json` 先提交至 `a1fac7a8` 后再启动盲测�?

---

## 3. 主检验与 2×2 探索格结果：累计转角 $\pi/2$ 边界分离机制

在保持每�?`SNR = 2 dB, σ = 0.10`、每条母波形评估 `(±ω) × 3 种旋�?× 3 套种子` 下，完整 `2×2`（单步频偏幅�?`|ω|` × 累计转角 `|ω|(h+�?`）盲测结果如下表（同步报告累计扩散方�?`σ²t`）：

| 盲测�?| 用�?| `|ω|` | `(h, �? t_p)` | 累计转角 `|ω|(h+�?` | 累计扩散 `σ²t` (`t=h+ℓ`) | 区组 / 母波�?| `phase` Acc / NLL | `zero` Acc / NLL | 配对准确率增�?(`95% t-CI` / `Boot`) | 配对 NLL 改善 (`95% t-CI` / `Boot`) | 同母波形 `s17` 物理 / `full_iq` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| **`G1_main_extrap_both`** | **主检验（双外推）** | `0.40` | `(2, 2, 4)` | **`1.60 rad`** | `0.04` | `131 / 8,384` | `23.80% / 1.4452` | `19.41% / 1.4809` | **`+4.39 pp`** `[+4.11, +4.68]` (`[+4.11, +4.67]`) | **`+0.0357`** `[+0.0330, +0.0383]` (`[+0.0330, +0.0383]`) | `64.62% / 0.9411` (`64.64% / 0.9290`) |
| **`G2_interp_both`** | 探索（双内插�?| `0.20` | `(2, 2, 4)` | **`0.80 rad`** | `0.04` | `8 / 512` | `33.88% / 1.3358` | `36.36% / 1.3081` | **`-2.48 pp`** `[-4.39, -0.58]` (`[-3.79, -0.89]`) | **`-0.0278`** `[-0.0459, -0.0096]` (`[-0.0414, -0.0134]`) | `46.78% / 1.0222` (`46.78% / 1.0193`) |
| **`G3_extrap_angle`** | 探索（仅外推转角�?| `0.20` | `(4, 4, 8)` | **`1.60 rad`** | **`0.08`** | `8 / 512` | `22.68% / 1.4410` | `19.74% / 1.4781` | **`+2.94 pp`** `[+1.73, +4.15]` (`[+2.03, +3.88]`) | **`+0.0371`** `[+0.0230, +0.0512]` (`[+0.0262, +0.0479]`) | `62.11% / 0.9796` (`61.91% / 0.9539`) |
| **`G4_extrap_freq`** | 探索（仅外推频率�?| `0.40` | `(1, 1, 2)` | **`0.80 rad`** | **`0.02`** | `8 / 512` | `33.29% / 1.3347` | `34.96% / 1.3098` | **`-1.67 pp`** `[-3.41, +0.06]` (`[-2.97, -0.34]`) | **`-0.0249`** `[-0.0339, -0.0160]` (`[-0.0320, -0.0184]`) | `47.75% / 1.0159` (`49.22% / 1.0037`) |

> **关键发现——决定性能坍塌的是累计转角是否越过 $\pi/2$ 象限边界，而非单步频偏幅度 `|ω|`**�?
> 1. 比较 `G4`（`|ω|=0.40` 但累计转�?`0.80 rad < π/2`）与 `G2`（`|ω|=0.20`, `0.80 rad`）：即便 `G4` 频率外推至未见的 `0.40`，两臂准确率仍维�?`33.29% / 34.96%`（均远超随机基线 `25%`），说明连接层与宿主并未被文字上�?`0.40` 击穿；相反，由于训练中未见过大转角折返且 6 个额外连续输入维度带来了训练初期优化阻力，`zero` 在支持角内（`G2, G4`）反而以 `1.67�?.48 pp` �?`0.025�?.028 nats` 稳定领先 `phase`�?
> 2. 比较 `G3`（`|ω|=0.20` 已在训练范围内，�?`h=�?4` 使累计转角达 `1.60 rad > π/2`）与 `G1`（`|ω|=0.40`, `1.60 rad`）：只要累计转角跨过 $\pi/2 \approx 1.5708\text{ rad}$，即�?`|ω|=0.20` 本身落在训练分布内部（`G3`），两臂均立刻跌�?`22.68% / 19.74%`！而同期物理核�?`1.60 rad` 下反而因扇区分布恰好与离散转角对齐达到更高准确率（`62.11%�?4.62%`），哪�?`G3` 的累计扩�?`σ²t = 0.08` 达到 `G4` �?4 倍（导致 `G3` 物理 `full_iq` 略低�?`G1`：`61.91%` vs `64.64%`），真正支配两臂胜负反转的轴依然�?**$|ω|(h+\ell) = 0.80$ vs $1.60\text{ rad}$**�?

*注：补充核对 `G1..G4` 跨种子离散度——`G2` 准确率差跨种子范�?`[-6.05, +0.26] pp`（SD `3.24 pp`）、NLL 改善范围 `[-0.0436, -0.0166] nats`；`G3` 准确率差 `[+2.21, +3.71] pp`（SD `0.75 pp`）、NLL 改善 `[+0.0133, +0.0542] nats`；`G4` 亦与 `G2` 呈现相同内插方向模式�?

---

## 4. 机制诊断 I：Work F 频偏符号干预与单模态矛盾探�?

### 4.1 固定缓存四格干预汇总（主格 `±0.40` 与历史配�?`±0.35`�?

| 实验配置 / 系统 | 方向增益 (`gain`) | 残差�?(`resid`) | 平均符号 `TV` | 旋转 `(1,0)` `TV` | 匹配 Acc / NLL | 交换 Acc / NLL | 交换减匹�?NLL (`[95% t-CI]`) | 因子均方范数 $(M^2, W^2, I^2)$ |
|---|---:|---:|---:|---:|---:|---:|---|---|
| **`±0.40` 物理校准参考（131 区块�?* | `1.0000` | `0.0000` | `0.6943` | `0.0000` | `64.62% / 0.9413` | `3.63% / 2.9964` | **`+2.0551`** `[+2.0252, +2.0850]` | `(0.2016, 2.3e-5, 2.8e-5)` |
| **`±0.40` `phase`（种子均�?/ 均值表�?* | **`+0.0098`** | `0.0549` (`0.0338`) | `0.0400` (`0.0257`) | `0.0831`（表�?| `23.74%` (`18.19%`) | `20.77%` (`18.01%`) | `+0.0184`（种子均值）/ **`+0.0162`** `[+0.0152, +0.0171]`（均值表�?| `(1.85e-4, 1.12e-3, 6.36e-5)` |
| **`±0.40` `zero`（种子均�?/ 均值表�?* | **`+0.0050`** | `0.0410` (`0.0260`) | `0.0279` (`0.0184`) | `0.0981`（表�?| `19.42%` (`18.17%`) | `19.87%` (`18.06%`) | `+0.0074`（种子均值）/ **`+0.0080`** `[+0.0073, +0.0086]`（均值表�?| `(9.96e-5, 1.02e-3, 4.33e-5)` |
| **`±0.35` 物理校准参考（16 区块�?* | `1.0000` | `0.0000` | `0.6774` | `0.0000` | `63.77% / 0.9516` | `3.08% / 2.8830` | **`+1.9313`** `[+1.8255, +2.0372]` | `(0.1871, 3.56e-5, 2.49e-5)` |
| **`±0.35` `phase`（独�?`check_f.py` 输出�?* | **`+0.0111`** | `0.0369` | `0.0267` | `0.0853` | `22.39%` | `21.01%` | **`+0.0181`** `[+0.0167, +0.0195]`�?000-boot�?| `(2.19e-4, 1.08e-3, 6.27e-5)` |
| **`±0.35` `zero`（独�?`check_f.py` 输出�?* | **`+0.0056`** | `0.0267` | `0.0192` | `0.0991` | `21.58%` | `21.61%` | **`+0.0093`** `[+0.0084, +0.0103]`�?000-boot�?| `(1.06e-4, 1.03e-3, 3.99e-5)` |

- **历史 `±0.35` 正世�?lookup 锚点验证**：Stage 3B [c4_s17_lookup.json](../stage3b_math/A/c4_s17_lookup.json) �?`H_c4_035` `+0.35` 正世界（16 区块 = `3,072` 条）实测 **`63.18% / 0.9651`**，与本轮独立重新校准物理参考在 `+0.35` 正世界下逐位相等（`63.18359375% / 0.9662`）�?
- **按各种子细察**：在 `±0.40` �?`phase` 三套种子的方向增益分别为 `-0.0015 (s202611), +0.0210 (s202622), +0.0099 (s202633)`；`zero` �?`+0.0018, +0.0129, +0.0004`。此外在均方效应量上�?W^2 \approx 0.00112$ 远超 $M^2 \approx 0.000185$（物理参考为 $M^2 = 0.2016$），表明**宿主仍几乎只按编码波形状态而非频偏符号转移概率**；但即便微弱，`phase` 臂的元数据响�?$M^2$ 确实约为 `zero` 臂的 **1.9 �?*�?

### 4.2 具体真实单缓存样本原始分数检验（`G1`, `seed = 202611`, `mother_idx = 3153`, `cache_code = 100`, `z1 = 6, z2 = 4`�?

物理参考在真实缓存 `(z1=6, z2=4)` 上对负正频偏分别给出截然相反的主导判断（负世�?`cal_ref_neg` 强烈指向 **扇区 0: `0.5660`**，`[0.5660, 0.2877, 0.0495, 0.0968]`；正世界 `cal_ref_pos` 强烈指向 **扇区 2: `0.5636`**，`[0.0520, 0.0845, 0.5636, 0.2999]`）。真实冻结宿主在同一缓存上的四候�?*未归一化原始序列对数似�?(`raw_loglik`)** 与温度校准后四概率如下：

| �?/ 告知频偏 | 四候选原始序列对数似�?`[s0, s1, s2, s3]` | 校准后四概率 `[s0, s1, s2, s3]` | 宿主 argmax 判决 | 判断解释 |
|---|---|---|---|---|
| **`phase` 告知 `ω = -0.40`** | `[-1.5578, -1.5616, -1.4460, -1.4353]` | `[0.2218, 0.2201, 0.2761, 0.2820]` | **扇区 3** | 从正频偏改为负频偏后，扇�?0 似然�?`-1.4337` 降至 `-1.5578`，改判扇�?3 |
| **`phase` 告知 `ω = +0.40`** | `[-1.4337, -1.5660, -1.5834, -1.5640]` | `[0.3038, 0.2344, 0.2265, 0.2353]` | **扇区 0** | **恰好做出与正世界真实最优（扇区 2: `0.5636`）对角反指（扇区 0）的判决**——直观印证了超越 $\pi/2$ 后未掌握反折符号旋转的失配本�?|
| **`zero` 告知 `ω = -0.40`** | `[-1.8306, -1.6399, -1.2379, -1.3382]` | `[0.1795, 0.2155, 0.3171, 0.2879]` | **扇区 2** | 同样反判到正世界扇区 2 |
| **`zero` 告知 `ω = +0.40`** | `[-1.7919, -1.6160, -1.2035, -1.4235]` | `[0.1863, 0.2206, 0.3278, 0.2653]` | **扇区 2** | 更换告知符号几乎不改判别（`prob_2` �?`0.3171 �?0.3278`），基本忽略文本中符号翻�?|

### 4.3 `phase` 臂单模态矛盾探针（`G1`, `|ω| = 0.40`�?

| 矛盾条件 | 平均准确�?| 平均 NLL | 相对双一致匹�?`TV` | 相对双反向交�?`TV` | 机制解读 |
|---|---:|---:|---:|---:|---|
| 仅翻转文本符号、保留真实相位槽 | `22.45%` | `1.4485` | `0.0281` | `0.0372` | 概率更贴近纯相位匹配端（变化 `0.0281 < 0.0372`�?|
| 仅翻�?6 个相位槽、保留真实文本符�?| `21.67%` | `1.4623` | **`0.0371`** | `0.0281` | **仅翻�?6 个数值槽即可将分布推向双交换端，引起更大的概率偏移（`0.0371`）与 NLL 劣化（`+0.0171` 相对 `1.4452`），证明连续槽在权重上略占主导、文本符号仅贡献次级调制** |

---

## 5. 机制诊断 II：Work G 旋转等变与离线外部群平均分离报告

| 诊断维度（主�?`G1` `±0.40`�?31 区组�?| 相位�?`phase` | 对照�?`zero` |
|---|---|---|
| **原始宿主：`(0,1)` 仅转第二窗保持�?*（`mean / worst max_abs`；`>0.20` 占比�?| `0.0591 / 0.4817`；`5.11%` | `0.0620 / 0.3587`；`5.80%` |
| **原始宿主：`(1,1)` 两窗同转一象限循环移位**（`mean / worst max_abs`；`>0.20` 占比�?| `0.1263 / 0.3800`；`15.44%` | `0.1334 / 0.3792`；`18.96%` |
| **原始宿主：`(1,0)` 仅转第一窗循环移�?*（`mean / worst max_abs`；`>0.20` 占比�?| `0.1237 / 0.3752`；`15.13%` | `0.1316 / 0.3859`；`17.93%` |
| **最�?`(1,1)` 逐例违例快照** | `s202622, +ω, mother=39, c00=198 �?c11=10`；`probs_00=[0.1395, 0.1004, 0.1689, 0.5913]` vs `probs_11=[0.2113, 0.3310, 0.2750, 0.1827]`（`diff = 0.3800`�?| `s202611, +ω, mother=1637, c00=9 �?c11=77`；`probs_00=[0.6403, 0.0815, 0.0630, 0.2152]` vs `probs_11=[0.2778, 0.2611, 0.2501, 0.2110]`（`diff = 0.3792`�?|
| **原始宿主对照：`independent_x3` 独立第三符号**（应接近均匀 `25%`�?| `Acc = 24.91%`, `NLL = 1.4237`, `mean_tv_from_unif = 0.0947`, `max_tv = 0.2935` | `Acc = 24.97%`, `NLL = 1.4315`, `mean_tv_from_unif = 0.1122`, `max_tv = 0.2783` |
| **原始宿主对照：`shuffled_Z` 打乱缓存** | `Acc = 25.20%`, `NLL = 1.4601` | `Acc = 25.16%`, `NLL = 1.5070` |
| **原始宿主总体指标（未做任何离线对称平均）** | **`23.80% / 1.4452`** | **`19.41% / 1.4809`** |
| **离线外部对称操作 I�?6 路群平均 `Pf`**（输出后 CPU 计算�?| `17.05% / 1.4113`（相�?raw `Acc -6.75 pp [-7.09, -6.41]`，`NLL +0.0339`�?| `16.43% / 1.4391`（相�?raw `Acc -2.98 pp [-3.29, -2.66]`，`NLL +0.0418`�?|
| **离线外部对称操作 II：单代表 `F0`**（输出后 CPU 计算�?| `20.88% / 1.4112` | `20.14% / 1.5090` |

> **解读**：由�?Qwen 连接层在训练中只采用软旋转不变提示而不是严�?$\mathbb{Z}_4 \times \mathbb{Z}_4$ 群等变连接，原始宿主仍存在约 `0.06` (`rot01`) �?`0.12�?.13` (`rot11/10`) 的等变破缺。有趣的是，当原始准确率本身围绕或略低于 `25%` 且存在跨象限系统误判时，离线强制执行 16 路群平均 `Pf` 虽能把极化预测拉平滑以降�?NLL（`1.4452 �?1.4113`），反而使准确率进一步下降至 `17.05%`——说明外部群平均并不能凭空创造未学到�?$\pi/2$ 物理方向知识，二者必须在报告与统计中严格分列�?

---

## 6. 机制诊断 III：Work A 迟到导频 `C4 (ω = +0.35, t_p = 4)` 七臂对比与旧工况回放

### 6.1 Q2 真实输入表示与连续参考边界说�?

- **输给宿主的导频表�?*：保�?Stage 3 �?12 维连续特征向�?`build_ref_feature(r)`（实部与虚部匹配投影及其归一/离散辅助统计，通过 `ref_mlp` 映射�?4 �?`num_ref_tokens` 连续隐层 token 插入题面对应 `<REF>` 槽位），**未对导频再做 `s17` 4-bit 离散量化**；同时仅�?`(z1, z2)` 两主数据窗保�?`s17` 4-bit 量化。因此，理论连续参考（�?`full_iq`）比 `q2_reader.read` 离散核能看到两窗与导频间未经 4-bit 离散截断的精确连续幅相�?
- **前端去重实现**：由 [data_stage3d.py](data_stage3d.py) `build_q2_arm_references()` 提供共享 `deduplicate` 标志。`duplicate_copy_dedup` �?`valid` �?`acquisition_id` 去重后产�?100% 相同�?`<REF>` token 与提示词；`duplicate_copy_raw` 刻意关闭去重以测量宿主对重复 `<REF>` token 的冗余敏感度�?

### 6.2 Work A 七臂与物理参照表�?12 条母波形�? 种子汇�?vs CPU `q2_reader.read`�?

| Q2 证据�?| `phase` Acc / NLL (`ECE`) | `zero` Acc / NLL (`ECE`) | `q2_reader.read` 物理�?Acc / NLL (`ECE`) | 相对基线物理/行为意义 |
|---|---:|---:|---:|---|
| `none`（无参考） | `24.80% / 1.3928` (`0.0230`) | `25.78% / 1.3938` (`0.0189`) | `25.00% / 1.3863` (`0.0000`) | 盲先验基�?|
| `valid`�? 条同信道导频�?| **`28.78%`** / `1.3889` (`0.0304`) | `26.63% / 1.4107` (`0.0557`) | `66.41% / 0.8192` (`0.0301`) | 有效证据进入 |
| `duplicate_copy_dedup`（副本去重） | **`28.78%`** / `1.3889` (`0.0304`) | `26.63% / 1.4107` (`0.0557`) | `66.41% / 0.8192` (`0.0301`) | **�?`valid` 逐例零差�?*（`mean_tv = 0.0`�?|
| `duplicate_copy_raw`（副本未去重�?| `26.30% / 1.3944` (`0.0284`) | `25.78% / 1.4133` (`0.0463`) | �?| 仅重复提示同条采集向量；相对 `valid`: `phase -2.47 pp` (`TV=0.0404`, `160` 改对 / `198` �?, `zero -0.85 pp` (`TV=0.0421`) |
| `independent_only`（纯独立无关�?| `24.80% / 1.4077` (`0.0364`) | `24.93% / 1.4205` (`0.0499`) | `25.00% / 1.3863` (`0.0000`) | 相对 `none`: `phase +0.00 pp` (`TV=0.0423`, `222` �?/ `222` �?, `zero -0.85 pp` (`TV=0.0663`) |
| `valid_plus_independent` | `24.54% / 1.4082` (`0.0425`) | `27.28% / 1.4183` (`0.0480`) | `66.41% / 0.8192` (`0.0301`) | 无效导频干扰；相�?`valid`: `phase -4.23 pp` (`TV=0.0403`, `159` �?/ `224` �?, `zero +0.65 pp` (`TV=0.0681`) |
| **`fresh_resample`**（第 2 次独立采样） | `26.43% / 1.4056` (`0.0348`) | `25.78% / 1.4064` (`0.0592`) | **`74.22% / 0.6632`** (`0.0333`) | 相对 `valid`：物理核 **`+7.81 pp / -0.1560 nats`**（Stage 3C 回归基准 `+8.5636 pp`）；**而宿�?`phase` �?`-2.34 pp / +0.0168 nats`（`172` �?/ `208` �? `max_conf_inc=0.4181`），`zero` �?`-0.85 pp`（`159` �?/ `172` 错）** |

- **具体母波形轨迹例（`mother_idx = 36, z1 = 11, z2 = 15, true_x1 = 3`, `seed = 202611`�?*：`q2_reader.read` 物理核随着有效参考由 `none [0.25, 0.25, 0.25, 0.25]` �?`valid [0.0100, 0.2563, 0.6672, 0.0665]`（尚偏向索引 2）→ `fresh_resample [0.0201, 0.0217, 0.2945, 0.6638]` 准确将真标签 `3` 概率�?`0.0665` 拉升�?**`0.6638`** 扭转判决；而同一母波形上 `phase` 臂输出几乎停滞在 `[0.2497, 0.2495, 0.2503, 0.2505]`（`valid`）与 `[0.2498, 0.2494, 0.2503, 0.2505]`（`fresh_resample`），`zero` 臂虽�?`fresh_resample` �?`prob_3` 升至 `0.2964`，整体却不敌其他样本上的置信扰动导致全样本净�?`-0.85 pp`�?

### 6.3 旧工�?`C1 / C2 / C3` 回放未遗忘验证（`s3_test` × 2 区块 × 3 旋转 = 384 �?× 3 种子�?

| 条件 | 任务 | `phase` Acc / NLL | `zero` Acc / NLL | 同期 `s17` 物理读出 Acc / NLL |
|---|---|---:|---:|---:|
| `C1_coherent_2db` | `Q1 / Q3 / Q2_valid` | `26.48%/1.3764` · **`39.84%/1.2510`** · `36.07%/1.3333` | `29.86%/1.3724` · **`50.09%/1.1876`** · `44.01%/1.2889` | `67.19%/0.8332` · `64.84%/0.9243` · `67.97%/0.8781` |
| `C2_discrete_walk` | `Q1 / Q3 / Q2_valid` | `30.90%/1.3652` · `36.37%/1.3399` · `28.47%/1.3815` | `28.13%/1.3755` · `38.80%/1.3055` · `32.20%/1.3602` | `61.72%/1.0057` · `53.13%/1.1671` · `46.88%/1.1928` |
| `C3_mid_pilot_cont` | `Q1 / Q3 / Q2_valid` | `29.77%/1.3700` · `35.16%/1.3003` · `35.59%/1.3324` | `30.99%/1.3715` · `42.62%/1.2437` · `39.15%/1.3019` | `60.16%/0.9926` · `60.94%/1.0450` · `63.28%/0.9177` |

可见回放集未崩溃，且�?`G2/G4` 一致：`zero` 臂在多任务支持集拟合�?`phase` 更深（如 `C1 Q3: 50.09%` vs `39.84%`），再次印证 **6 个连续数值槽在有�?1,280 步微调内略微分散了标准工况上的特征对�?*�?

---

## 7. 历史 Ministral-3-3B 诊断状态、范围边界、复算与核心后续观测

### 7.1 实际完成与未完成项目透明清单

- **已全部完成并核验**：Qwen3.5-4B 完整 Smoke �?Tokenizer 审计、显存与引用环消除基准测试�? 种子 × 2 臂（6 次）配对训练与开发集温度校准、`dev_pilot`（`B_locked=131`）与提交前锁定（`Commit #1`）、全量配对盲测（`G1` + `G2..G4` + 历史 `±0.35`）、Work F（含单模态矛盾探针与 `check_f.py`）、Work G（含离线 `Pf/F0` 严格分离）、Work A Q2 七臂�?`q2_reader.read` 对照、`C1/C2/C3` 遗忘回放、`MANIFEST.sha256` 及干净目录零误差全量重算验证（`max_abs_diff = 0.000e+00`）�?
- **未完成项（按严格指纹锁定协议主动中止�?*：计划末段可选的历史 `mistralai/Ministral-3-3B-Instruct-2512-BF16` 旧检查点复验（`modal_runner.py --mode eval_ministral`, Modal App `ap-ILaRaUzoj3xPA12N7F3Dsc`）已完成锁定 Hugging Face revision `b6d637bef2393152b3da2b2fde72eecdee30557e` 下载及权重读取，但在执行 `assert host.host_sha256 == lock["host_sha256"]` 时按协议主动终止报错——因 Qwen3.5-4B 所需的容器环境（`transformers 5.24.1 + torch 2.15.0+cu132`）与 Stage 3 训练历史 Ministral 时锁定的环境（`transformers 5.22.1 + torch 2.14.1+cu130`）在采样指纹逻辑上不一致（预期参数名称/采样格式跨版变更）。为严守不放松任何既有阶�?SHA-256 指纹锁、且不修�?`research/stage3_ministral_cache/` 既有代码的原则，未强制跳过指纹校验运行，故历�?Ministral 中文重放本轮列为“受运行库版本隔离限制而中止的未完成可选诊断”�?

### 7.2 适用范围与限制声�?

本轮全部实证结论严格限定于以下条件之内：
1. **指定物理模拟�?*：QPSK 四相脉冲、`N_SAMPLES = 4`、`SNR = 2 dB`（及各条件固�?`d / σ_φ`）、连续布朗相位随机游走加线性频偏�?
2. **冻结组件版本**：`Qwen/Qwen3.5-4B`（`851bf6e8...`）、`equivariant_b8_s17`（`e67d9fba...`）与本轮 6 组连接层（每组仅 `2,184,964` 可训练参数）�?
3. **训练预算与分�?*：每连接�?`8 epochs × 160 steps`（`batch_size = 32`, `lr = 1.5e-3`, `kv_cached` 评分器）、支持域 $|\omega|(h+\ell) \le 1.25\text{ rad}$ 与本文盲测网格分布；更高参数量微调、更长步数或多阶相位直接耦合结构上的行为不在本轮已验证范围之内�?

### 7.3 独立一行复算命令（已于干净临时目录验证通过�?

�?[recompute_all.py](recompute_all.py) 与紧凑原始评分包（`results/{blind_q3_scores.npz, blind_q2_scores.npz, replay_c123_scores.npz, summary_master.json}`，仅依赖标准�?+ `numpy==2.3.5` + `scipy==1.17.0`，零 PyTorch 与仓内模块导入）拷贝至任意干净目录后执行：

```bash
uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B recompute_all.py --data-dir results --verify-against results/summary_master.json
# 实测�?%TEMP%\stage3d_clean_verify 输出：VERIFIED OK: max_abs_diff=0.000e+00，退出码 0
```

如需在本仓库执行基线及回归复验：

```bash
uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B research/stage3c/verify.py --repo .
uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 --with torch --with pytest python -B -m pytest research/stage3d_qwen_signed_phase/tests/test_stage3d_cpu.py -v
```

### 7.4 本轮最值得继续追查的核心观�?

> **最值得优先追查的观测：为何跨过累计转角 $\pi/2$ 边界�?|ω|(h+\ell) = 0.80 \rightarrow 1.60\text{ rad}$）后两臂均出现急剧断崖，且当前加性连�?`<PHASE>` token�?,280 步）只诱导出 `+0.0098`（物理参考约 1%）的微弱方向增益�?*
> - 实验揭示，只要把 `|ω|` 提高�?`0.40` 但缩短延迟使总转角留�?`0.80 rad`（`G4`），宿主的准确率就能维持�?`33.3% / 35.0%`；而一旦保持已知频�?`|ω| = 0.20` 仅拉长延迟使总转角越�?$\pi/2$ 到达 `1.60 rad`（`G3`），宿主立即跌回 `22.7% / 19.7%`，甚至在典型单缓存样例上给出与真值完全对角反指（扇区 0 vs 扇区 2）的自信判决�?
> - 原因直指结构感应偏置�?*�?`[cos(ωt), sin(ωt)]` 经过独立 MLP 编码�?3 个追加在序列尾部的加�?`<PHASE>` �?token，仍要求冻结语言模型自己通过跨注�?线性状态衰减去“手算”连续三维旋转对离散 `(z1, z2)` 表征的调制，这既拖慢了支持域内的收敛（解释了为何 `G2/G4` �?`phase` 落后 `zero` �?`2 pp`），又无法保�?$\text{SO}(2)$ 等变群外推�?* 后续最自然也最干净的对照实验是：将 `[cos(ωt), sin(ωt)]` 从外挂加�?token 升级为连接层内部�?*显式低维双线�?复旋转调�?*（在传入冻结宿主之前将编码器特征按公开相位角预先旋转或与相位基做群协变耦合，或将训练支持稍微跨�?$\pi/2$ �?$|\omega|(h+\ell) \approx 1.8$ 区分“表示欠约束”与“支持未覆盖”）�?

---

## 交付清单与指纹锚�?

- 锁定协议：[PROTOCOL_LOCK.json](PROTOCOL_LOCK.json) · 盲测区组清单：[blind_seed_manifest.json](blind_seed_manifest.json)
- 完整检验汇总：[results/summary_master.json](results/summary_master.json) · 干净目录复算脚本：[recompute_all.py](recompute_all.py)
- 全部交付物完整性校验清单：[MANIFEST.sha256](MANIFEST.sha256)
- **连接�?Weights 与原始分数包**：全�?6 套连接层 checkpoint（[checkpoints/connector_*.pt](checkpoints/)，单文件�?`8.75 MB`，参�?Stage 3 仓库惯例直接提交于分支并镜像保存�?Modal �?`stage3d-results:/vol/stage3d/checkpoints/`）以及轻�?`.npz` 原始分数包（共约 `4.0 MB`）已随分�?`research/stage3d-qwen-signed-phase` 完整提交�?GitHub，逐文件精�?SHA-256 锁定�?[PROTOCOL_LOCK.json](PROTOCOL_LOCK.json) �?[MANIFEST.sha256](MANIFEST.sha256)，克隆后即可�?GPU 复算全表�?
````

### Event #11: `item.completed` — `web_search` (`exec-b999af51-4d6e-41dd-98d7-b75711c01b1b`)

```json
{
  "id": "exec-b999af51-4d6e-41dd-98d7-b75711c01b1b",
  "type": "web_search",
  "query": "site:modal.com/docs volumes pricing monthly free CPU memory A100 80GB L4 A10 Modal spending limit starter ...",
  "action": {
    "type": "search",
    "queries": [
      "site:modal.com/docs volumes pricing monthly free CPU memory A100 80GB L4 A10 Modal spending limit starter",
      "Mendeley k72b4pnc5g QPSKlink 822762822 Zenodo 4895133 CTTC dataset"
    ]
  },
  "results": [
    {
      "type": "text_result",
      "domain": "modal.com",
      "ref_id": "turn0search0",
      "snippet": "On Modal, a 3x multiplier is applied to the CPU and Memory list price for non-preemptible resources. ... Modal's serverless pricing can't be directly compared",
      "title": "Pricing: Pay per second for GPUs and CPUs | Modal",
      "url": "https://modal.com/pricing"
    },
    {
      "type": "text_result",
      "domain": "zenodo.org",
      "ref_id": "turn0search1",
      "snippet": "Centre Tecnològic de Telecomunicacions de Catalunya (CTTC) ... This is a simple dataset of raw IQ measurements on a point-to-point wireless indoor channel, captured in",
      "title": "Datasets of Indoor Wireless Channel Measurements for Machine Learning Applications",
      "url": "https://zenodo.org/records/4895133"
    },
    {
      "type": "text_result",
      "domain": "modal.com",
      "ref_id": "turn0search2",
      "snippet": "Nvidia A100, 80 GB ... Nvidia L4 ... * Volume-based discounts ... Graduate students, labs, and researchers can get up to $10k free compute credits",
      "title": "Plan Pricing | Modal",
      "url": "https://modal.com/pricing?trk=public_post-text"
    },
    {
      "type": "text_result",
      "domain": "data.mendeley.com",
      "ref_id": "turn0search3",
      "snippet": "Published: 3 June 2026| Version 1 | DOI: 10.17632/k72b4pnc5g.1 ... To access this dataset in MATLAB, copy the file in your working folder and then",
      "title": "Frames of SDR ADALM PLUTO based QPSK Communication Link",
      "url": "https://data.mendeley.com/datasets/k72b4pnc5g/1"
    },
    {
      "type": "text_result",
      "domain": "modal.com",
      "ref_id": "turn0search4",
      "snippet": "Up-to-date unit pricing for all of our products is available on our Pricing page. ... At the end of each billing cycle, you are auto-charged",
      "title": "Billing | Modal Docs",
      "url": "https://modal.com/docs/guide/billing"
    },
    {
      "type": "text_result",
      "domain": "modal.com",
      "ref_id": "turn0search5",
      "snippet": "Volumes are a high-performance distributed file system for Modal applications. ... The easiest way to create a Volume and use it as a part of",
      "title": "Volumes | Modal Docs",
      "url": "https://modal.com/docs/guide/volumes"
    },
    {
      "type": "text_result",
      "domain": "modal.com",
      "ref_id": "turn0search6",
      "snippet": "Each Modal Function or Sandbox container has a default request of 0.125 CPU cores and 128 MiB of memory. ... cpu_limit = 4.0 ... Modal",
      "title": "Configuring CPU, memory, and disk | Modal Docs",
      "url": "https://modal.com/docs/guide/resources"
    },
    {
      "type": "text_result",
      "domain": "modal.com",
      "ref_id": "turn0search7",
      "snippet": "Currently B300, B200, H200, H100, A100, L4, T4 and L40S instances support up to 8 GPUs (up to 2,304 GB GPU RAM), and A10 instances",
      "title": "GPU acceleration | Modal Docs",
      "url": "https://modal.com/docs/guide/gpu"
    },
    {
      "type": "text_result",
      "domain": "modal.com",
      "ref_id": "turn0search8",
      "snippet": "* Use a Workspace budget when you want one overall monthly cap for the Workspace. ... If you do not set a custom spend limit,",
      "title": "Budgets | Modal Docs",
      "url": "https://modal.com/docs/guide/budgets"
    },
    {
      "type": "text_result",
      "domain": "modal.com",
      "ref_id": "turn0search9",
      "snippet": "## Sign up for Modal Get started with $30 free monthly compute!",
      "title": "Sign up | Modal",
      "url": "https://modal.com/signup"
    },
    {
      "type": "text_result",
      "domain": "interactca20120.org",
      "ref_id": "turn0search12",
      "snippet": "It is intended primarily for researchers without access to software defined radio equipment, and should be helpful for understanding and investigating fundamental properties of real",
      "title": "- This dataset consists of raw IQ measurements of a wireless indoor channel. It is intended primarily for researchers without access to software defined radio equipment, and should be helpful for understanding and investigating fundamental properties of real wireless channels (e.g., for channel modeling). more info in https://zenodo.org/record/4895133",
      "url": "https://interactca20120.org/wp-content/uploads/2023/06/Deliverable-H1.pdf"
    },
    {
      "type": "text_result",
      "domain": "dev.mendeley.com",
      "ref_id": "turn0search10",
      "snippet": "Once published, it can be referenced from other sources using its DOI, which is made up of the dataset’s unique identifier (e.g. `322y43x9b7`) and its",
      "title": "Datasets API Quick Start Guide Mendeley Developer Portal",
      "url": "https://dev.mendeley.com/code/datasets_quick_start_guides.html"
    },
    {
      "type": "text_result",
      "domain": "modal.com",
      "ref_id": "turn0search11",
      "snippet": "Modal's per-second pricing applies to GPU compute as well as CPU and memory resources, while Functions automatically scale to zero by default when idle. ...",
      "title": "Best Cloud Platforms to Rent A10 GPUs in 2026 | Modal",
      "url": "https://modal.com/resources/best-cloud-platforms-rent-a10-gpus"
    },
    {
      "type": "text_result",
      "domain": "ds.jpeg.org",
      "ref_id": "turn0search13",
      "snippet": "Reflecting the fact that the CfP asks for coding solutions developed using learning-based techniques, this CTTC document defines and provides access information to a training",
      "title": "wg1n100276-096-PCQ-Common_Training_and_Test_Conditions_for_JPEG_Pleno_Point_Cloud_v1_3",
      "url": "https://ds.jpeg.org/documents/jpegpleno/wg1n100276-096-PCQ-Common_Training_and_Test_Conditions_for_JPEG_Pleno_Point_Cloud_v1_3.pdf"
    },
    {
      "type": "text_result",
      "domain": "www.nature.com",
      "ref_id": "turn0search14",
      "snippet": "= 22, Roll-off factor = 0.35, SOQPSK: ρ = 0.7, B = 1.25, T₁ = 1.5, T₂ = 0.5,</td> ... <td>GMSK BT<sub>b</sub> product = 0.3,",
      "title": "Cognitive Radio for Satellite TT & C System: A General Dataset Using Software-defined Radio",
      "url": "https://www.nature.com/articles/s41597-026-07182-7.pdf"
    },
    {
      "type": "text_result",
      "domain": "dl.ifip.org",
      "ref_id": "turn0search15",
      "snippet": "Centre Tecnològic de Telecomunicacions de Catalunya (CTTC/CERCA) ... In [2], an experimental demonstration showcased the coexistence of CV-QKD and 7×12.5 Gbit/s on-off keying (OOK). In",
      "title": "SDN-Enabled Continuous-Variable QKD in Coexistence with 8×200 Gb/s 16-QAM Classical Channels",
      "url": "https://dl.ifip.org/db/conf/ondm2024/ondm2024/1570996377.pdf"
    },
    {
      "type": "text_result",
      "domain": "ondm2018.scss.tcd.ie",
      "ref_id": "turn0search16",
      "snippet": "The experimental setup consists of a transmitter (SS-STx) on the left, a transmission link in the center, and a receiver (SS-SRx) on the right. ...",
      "title": "ONDM2018_CTTC",
      "url": "https://ondm2018.scss.tcd.ie/wp-content/uploads/2018/05/Munoz_publish.pdf"
    },
    {
      "type": "text_result",
      "domain": "modal.com",
      "ref_id": "turn0view0",
      "snippet": "Total lines: 391",
      "thumbnail_url": "https://images.openai.com/static-rsc-1/b2paIXpQvr_aqPym5wKwLg7t6DZR_yg3e28hulHsGmdVte8uOeB0Kn69QaKlPmGDafsI33GzZo-Yd1idXsqXBg",
      "title": "Pricing: Pay per second for GPUs and CPUs | Modal",
      "url": "https://modal.com/pricing"
    },
    {
      "type": "text_result",
      "ref_id": "turn0view1",
      "title": "Internal Error"
    },
    {
      "type": "text_result",
      "ref_id": "turn0view2",
      "title": "Internal Error"
    },
    {
      "type": "text_result",
      "domain": "huggingface.co",
      "ref_id": "turn0view3",
      "snippet": "Total lines: 880",
      "thumbnail_url": "https://images.openai.com/static-rsc-1/yUn4H8IeGBBkNoboG-bEIZQNMQIK7RTkpyEy07VuWRpjDnCiqxOWZKw2N02j-54LmFbNM9Gv-aKMPmxs-UbpOKCZP9m2j9pJyTt4ZD97jY4ZTs6zHY8YL2q-TTpcGzRkrpgMGjikBPggrt1db5e4PA",
      "title": "Qwen/Qwen3.5-4B · Hugging Face",
      "url": "https://huggingface.co/Qwen/Qwen3.5-4B"
    }
  ]
}
```

### Event #12: `item.completed` — `web_search` (`exec-8584688a-e9ce-4857-b522-c9d5a67317d7`)

```json
{
  "id": "exec-8584688a-e9ce-4857-b522-c9d5a67317d7",
  "type": "web_search",
  "query": "https://zenodo.org/api/records/4895133/files",
  "action": {
    "type": "open_page",
    "url": "https://zenodo.org/api/records/4895133/files"
  },
  "results": [
    {
      "type": "text_result",
      "ref_id": "turn1view0",
      "title": "Internal Error"
    },
    {
      "type": "text_result",
      "ref_id": "turn1view1",
      "title": "Internal Error"
    },
    {
      "type": "text_result",
      "domain": "modal.com",
      "ref_id": "turn1view2",
      "snippet": "Total lines: 391",
      "thumbnail_url": "https://images.openai.com/static-rsc-1/Tr_JpQ6JFZIuLmKs7rsS9MZQfCF2nT1RXVWTeMyamDPOy_9qkEB_mh95WbcSCrQNWRymc1PZ6hJwZqKzgWKOgQ",
      "title": "Pricing: Pay per second for GPUs and CPUs | Modal",
      "url": "https://modal.com/pricing"
    },
    {
      "type": "text_result",
      "domain": "modal.com",
      "ref_id": "turn1view3",
      "snippet": "Total lines: 391",
      "thumbnail_url": "https://images.openai.com/static-rsc-1/cHPtGmfbZorkovTGLmDuDP2uLPTj-1mXkehX9HBKmlr4NMj39oQKUA-LXNcAGT3YZghKoJScYS_8xsjosSm_Nw",
      "title": "Pricing: Pay per second for GPUs and CPUs | Modal",
      "url": "https://modal.com/pricing"
    },
    {
      "type": "text_result",
      "domain": "modal.com",
      "ref_id": "turn1view4",
      "snippet": "Total lines: 51",
      "thumbnail_url": "https://images.openai.com/static-rsc-1/hhdPaOvvr0iGOxV6zyrbMqJRnwk9aJnK583UzdH6ZcTVukjwqcjlqszxLCKCec_5H-OOyu1cs42qbVF2UX-_BxCvxmVTTouaJ7UHMpWBoaieO74mTIBPYu8S56PNMdgu",
      "title": "Budgets | Modal Docs",
      "url": "https://modal.com/docs/guide/budgets"
    }
  ]
}
```

### Event #13: `item.completed` — `command_execution` (`item_10`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "[Console]::OutputEncoding=[Text.Encoding]::UTF8; rg -n -i 'stage 4|phase|实测' docs/LONG_TERM_RESEARCH_PLAN.md; Get-Content -Encoding utf8 research/stage3d_qwen_signed_phase/PROTOCOL_LOCK.json -TotalCount 175; rg -n '"'^[|].*(20261|975.7)|1280|23.88'"' research/stage3d_qwen_signed_phase/REPORT.md; Get-Content research/stage3_ministral_cache/schema.py; rg -n 'B3_mixture' research -g '*.py'"
```

**Aggregated Output:**

```text
�޷��������ԡ�������ģʽ��֧�ֺ������͵��������á�
����λ�� ��:2 �ַ�: 1
+ [Console]::OutputEncoding=[Text.Encoding]::UTF8; rg -n -i 'stage 4|ph ...
+ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : PropertySetterNotSupportedInConstrainedLanguage
 
5:**研究快照为 2026 年 11 月 22 日。** 本计划覆盖 GitHub 当时可见的主分支和 17 条研究分支。分支末端与报告列在文末。仓库审阅以各阶段报告、证明和实测数据说明为依据，本轮没有重训模型或重跑历史波形。时间按一位研究者约 6–12 个月估计，阶段以完成的研究成果推进。
30:|四条实测分支、三种数据来源        |435 MHz 两条分支使用同一份 742 帧 16-QAM OFDM 录制。138 帧和 240 帧是不同划分。Pluto 提供连续时域 QPSK，CTTC 提供多会话 QPSK。另有宽带 Wi-Fi CSI 文件的格式和重复性审计。                                                                                                           |在不同信号输入层上分别验证可回答的问题；划分时以会话或物理录制为单位。宽带数据盘点供将来选题。 [435 MHz OFDM](https://github.com/20171806011/EM-FloatFormer/blob/ba39dd5968dbaee7b7a226a9bc78d536278fdee8/research/realdata_sdr_ofdm/REPORT.md) · [压力实验](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/REPORT.md) · [Pluto](https://github.com/20171806011/EM-FloatFormer/blob/ade8333fb68126251a83e484b69b821b5d97bcea/research/realdata_pluto_qpsk/REPORT.md) · [CTTC](https://github.com/20171806011/EM-FloatFormer/blob/1887aee6de9649bc0dd8a4f05fe3c4542f3f6d77/research/realdata_cttc_qpsk/REPORT.md) · [宽带样本审计](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/DATASET_DISCOVERY.md)|
31:|11 月 22 日 A–E 会合     |A/B 核对后验读出并定位同一观测被重复利用的问题。C 的连续频偏精化在部分离格条件有益，在跳变上可能有损。D 给出相位跳变后可辨识量和误码下界。E 给出何时值得切换估计器的理论条件，实际 14 帧合成检验比固定连续方法多错 172 位。会合报告中的 435 MHz 数字是对历史逐帧记录的复算。                                                                             |优先延伸 D，借用 A/B 的证据核算与 P7 的预测—揭示设计。频偏精化和门控作为具体问题的工具。 [五组会合](https://github.com/20171806011/EM-FloatFormer/blob/e914befb2c03838c5e911e1af55c64230f1f42d6/research/abcde_convergence_2026_11_22/README.md) · [D 组证明](https://github.com/20171806011/EM-FloatFormer/blob/e914befb2c03838c5e911e1af55c64230f1f42d6/research/abcde_convergence_2026_11_22/D/PROOF.md) · [实测输入与复算](https://github.com/20171806011/EM-FloatFormer/blob/e914befb2c03838c5e911e1af55c64230f1f42d6/research/abcde_convergence_2026_11_22/experiments/historical/REAL_INPUTS.md)                                                                                                                                                                                                                                                                      |
33:**最直接的起点是 D 组的同一段波形。** 已知非零信道、均匀未编码 16\-QAM、七种等概率非零跳角、无噪无抖动时，已知跳点之后，两个符号的比值仍可精确计算，每个原始标签位的最优平均错误率却是 3/7。若跳点也隐藏，加入一枚保证落在跳变后的已知非零导频，整帧期望 BER 从 24\.7347% 降到 2\.0612%。发生跳变的确切时刻仍可能有歧义。这里已经可以对同一段信号连续追问，并逐个说明答案凭什么成立。[五组会合](https://github.com/20171806011/EM-FloatFormer/blob/e914befb2c03838c5e911e1af55c64230f1f42d6/research/abcde_convergence_2026_11_22/README.md) · [D 组证明](https://github.com/20171806011/EM-FloatFormer/blob/e914befb2c03838c5e911e1af55c64230f1f42d6/research/abcde_convergence_2026_11_22/D/PROOF.md) · [实测输入与复算](https://github.com/20171806011/EM-FloatFormer/blob/e914befb2c03838c5e911e1af55c64230f1f42d6/research/abcde_convergence_2026_11_22/experiments/historical/REAL_INPUTS.md)
52:|4 · 第 3–6 个月，在真实录制上兑现一个预测  |“在陌生的一次采集中，哪段已知波形是有效参考？揭示后，先前的预测应怎样修正？”      |先用已有 Pluto 或 CTTC 的已知序列遮住—揭示，A 次采集设参数，B 次采集检验。CTTC 可留整会话；Pluto 可检查连续时域窗；435 MHz 网格可检验同题的 OFDM 版本。对缺少真值的问题使用留出导频预测或公开序列定义答案。分别汇报自然录制、基于实测的人工干预与独立来源。交付能从原输入重放的至少一项配对外测。|
66:|435 MHz 16-QAM OFDM，同步、去循环前缀、傅里叶变换后的复数网格             |先在已有导频及明确可评分的数据位置上遮住—揭示，之后才加受控频偏或相位跳变。                            |这份数据可检验网格上的相位证据。历史 240 帧 B3 对比学习头，在原始工况 BER 为 9.1534% 对 10.6077%，在注入频偏工况为 26.7733% 对 14.4038%；11 月 22 日会合复算了历史表，未重新运行实测 I/Q 推理。|
86:|4 · 实测可行性  |查明 Pluto、CTTC 哪些参考序列、输入视图和会话可配对；435 MHz 记录可检哪类题|“问题—输入—真值—划分”表，加一个可复现样例标识|
89:第一次汇合围绕同一个波形对完成三件事：数学组列出能排除的解释，基准组给出精确概率，实测组指出同样的观测在真实记录里能否取得。三者有分歧就收窄假设。这一步可以把一个有趣的信号处理现象变成第一条可供独立检验的科学主张。
{
  "experiment": "EM-FloatFormer Stage 3D (Qwen3.5-4B signed phase vs zero arm)",
  "model_id": "Qwen/Qwen3.5-4B",
  "hf_sha": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
  "encoder_name": "equivariant_b8_s17",
  "encoder_sha256": "e67d9fbafb90791ab19549de7789798d818286d7f52d7ac566cc39df71e78356",
  "chat_template_sha256": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
  "thinking_mode": "disabled (enable_thinking=False -> <think>\\n\\n</think>\\n\\n)",
  "host_sha256": "f39c9e1b7a5719c1397dcdce42d221215fb20edadb881d2f81e745673636f64d",
  "versions": {
    "transformers": "5.24.1",
    "torch": "2.15.0+cu132",
    "numpy": "2.3.5",
    "scipy": "1.17.0",
    "gpu_name": "NVIDIA A100-SXM4-80GB"
  },
  "locked_hyperparameters": {
    "arm_names": [
      "phase",
      "zero"
    ],
    "connector_seeds": [
      202611,
      202622,
      202633
    ],
    "trainable_params": 2184964,
    "epochs": 8,
    "batch_size": 32,
    "micro_batch_size": 16,
    "lr": 0.0015,
    "max_steps_per_epoch": 160,
    "train_scorer": "kv_cached",
    "eval_scorer": "kv_cached",
    "eval_every_epochs": 2
  },
  "checkpoints": {
    "phase_s202611": {
      "filename": "connector_phase_s202611.pt",
      "checkpoint_sha256": "088d4ad34e1152ff000959a37bf784a8f45e326bdd2e483fdf755f82405f7866",
      "best_epoch": 7,
      "best_sel_score": 1.3809686587691041,
      "temp_q3": 0.5098932035670315,
      "temp_q2": 3.9999830531425227,
      "dev_q3_cal_nll": 1.375347147854539,
      "dev_q2_cal_nll": 1.3866422230228832,
      "host_sha256_after": "f39c9e1b7a5719c1397dcdce42d221215fb20edadb881d2f81e745673636f64d"
    },
    "phase_s202622": {
      "filename": "connector_phase_s202622.pt",
      "checkpoint_sha256": "ba275787201abf5f1837bd9ebcc959dab24746f9ffa1a08a5df93b0e8a263ae0",
      "best_epoch": 8,
      "best_sel_score": 1.338227526868415,
      "temp_q3": 0.8489407339257738,
      "temp_q2": 1.1069058926498243,
      "dev_q3_cal_nll": 1.285269302113726,
      "dev_q2_cal_nll": 1.3565327978349748,
      "host_sha256_after": "f39c9e1b7a5719c1397dcdce42d221215fb20edadb881d2f81e745673636f64d"
    },
    "phase_s202633": {
      "filename": "connector_phase_s202633.pt",
      "checkpoint_sha256": "75f0022894bde60b6e69673d670604b4f0a8cca7847d8df22339e36ba3ddd92f",
      "best_epoch": 8,
      "best_sel_score": 1.345658880106861,
      "temp_q3": 0.6574859852227614,
      "temp_q2": 1.0038161968358967,
      "dev_q3_cal_nll": 1.3001861761732214,
      "dev_q2_cal_nll": 1.3714957209402086,
      "host_sha256_after": "f39c9e1b7a5719c1397dcdce42d221215fb20edadb881d2f81e745673636f64d"
    },
    "zero_s202611": {
      "filename": "connector_zero_s202611.pt",
      "checkpoint_sha256": "3d09eded33b58b928191e54d344381ae141b70473d0409cd660d259ec742b77c",
      "best_epoch": 8,
      "best_sel_score": 1.360639128458629,
      "temp_q3": 1.0413709772961914,
      "temp_q2": 1.2209637372368802,
      "dev_q3_cal_nll": 1.3349880521106885,
      "dev_q2_cal_nll": 1.3793762964672136,
      "host_sha256_after": "f39c9e1b7a5719c1397dcdce42d221215fb20edadb881d2f81e745673636f64d"
    },
    "zero_s202622": {
      "filename": "connector_zero_s202622.pt",
      "checkpoint_sha256": "a8eb55bfad0d266cbf7e9dc9712b2d32b162374a39ab2898a7f54fa8724c3e98",
      "best_epoch": 8,
      "best_sel_score": 1.3124549620599506,
      "temp_q3": 0.7826950522516535,
      "temp_q2": 1.4118146530519748,
      "dev_q3_cal_nll": 1.233350441464049,
      "dev_q2_cal_nll": 1.3069782597576265,
      "host_sha256_after": "f39c9e1b7a5719c1397dcdce42d221215fb20edadb881d2f81e745673636f64d"
    },
    "zero_s202633": {
      "filename": "connector_zero_s202633.pt",
      "checkpoint_sha256": "ea7aebcc6d53bc7161f07ed38c0d5b21e473ff8b56aa3b5af1db61f80f9eb4b3",
      "best_epoch": 8,
      "best_sel_score": 1.341565335646975,
      "temp_q3": 0.6733062486614652,
      "temp_q2": 1.2734733247377854,
      "dev_q3_cal_nll": 1.292657943214399,
      "dev_q2_cal_nll": 1.3811923802166672,
      "host_sha256_after": "f39c9e1b7a5719c1397dcdce42d221215fb20edadb881d2f81e745673636f64d"
    }
  },
  "dev_pilot": {
    "namespace": 940202,
    "blocks": 16,
    "mothers": 1024,
    "pilot_block_seeds": [
      2338220280,
      1372146723,
      1251899241,
      1019832785,
      2632972314,
      1698518890,
      2353612898,
      1409965763,
      3844068644,
      2792309556,
      1248517795,
      579468612,
      575362531,
      3621001698,
      787505858,
      3687402811
    ],
    "observed_acc_diff_mean": -0.03352864583333334,
    "observed_acc_diff_block_sd": 0.014638601593173158,
    "observed_nll_diff_mean": -0.02437774790791658,
    "observed_nll_diff_block_sd": 0.013414584590285893,
    "chi2_upper_multiplier_df15_alpha05": 1.437305501981677,
    "conservative_upper_sd_hit": 0.02104014261118552,
    "conservative_upper_sd_nll": 0.019280856238416536,
    "work_e_scenario": {
      "q": 0.35,
      "rho": 0.01,
      "nll_mother_sd": 0.5,
      "pilot_block_sd": [
        0.02104014,
        0.01928086
      ],
      "sd": {
        "accuracy": 0.09441431300390846,
        "nll": 0.07979465834252315
      },
      "normal_by_endpoint": {
        "accuracy": 129,
        "nll": 52
      },
      "t_by_endpoint": {
        "accuracy": 131,
        "nll": 54
      },
      "normal_blocks": 129,
      "normal_mothers": 8256,
      "t_blocks": 131,
      "t_mothers": 8384,
      "normal_at_normal": {
        "accuracy": {
          "power": 0.9504225431012328,
          "plugin_half_width": 0.016292625434745873,
          "expected_half_width": 0.016292625434745873
        },
        "nll": {
          "power": 0.999905603276502,
          "plugin_half_width": 0.013769781706873525,
          "expected_half_width": 0.013769781706873525
        }
      },
      "t_at_normal": {
        "accuracy": {
          "power": 0.9475722738758099,
          "plugin_half_width": 0.016448130307161433,
          "expected_half_width": 0.016416036730692727
        },
28:| `202611` | `26.59% / 1.3950` (`0.0625`) | `19.98% / 1.4436` (`0.1274`) | `+6.62 pp` (`[+6.12, +7.11] pp`) | `+0.0486 nats` (`[+0.0457, +0.0516] nats`) |
73:   - **实测验证**：在最长输入组 `Q3_after_pilot`（`351` 字符 + `1` 条参考）与 `Q2_fresh_resample`（`2` 条参考）上连跑 4 个微批，分配显存严格恒定为 **`[8055.2, 8055.2, 8055.2, 8055.2] MB`**（零泄漏）；`micro_batch_size=16` 训练峰值稳定在 **`52,972.5 MB`**（稳态 `1,975.7 ms / 32-ex step`），盲测推理 `b=64` 峰值 **`45,409.9 MB`**（`23.88 ms/例`）。
104:| `phase / 202611` | `7` | `1.38097` | `1.37762` (`29.62%`) | `1.37535` (`0.5099`) | `1.38664` (`4.0000`) | `2884.0 s` / `52,972.5 MB` | `088d4ad34e1152ff` |
107:| `zero / 202611` | `8` | `1.36064` | `1.34627` (`35.68%`) | `1.33499` (`1.0414`) | `1.37938` (`1.2210`) | `2908.7 s` / `52,972.5 MB` | `3d09eded33b58b92` |
184:| **最坏 `(1,1)` 逐例违例快照** | `s202622, +ω, mother=39, c00=198 → c11=10`；`probs_00=[0.1395, 0.1004, 0.1689, 0.5913]` vs `probs_11=[0.2113, 0.3310, 0.2750, 0.1827]`（`diff = 0.3800`） | `s202611, +ω, mother=1637, c00=9 → c11=77`；`probs_00=[0.6403, 0.0815, 0.0630, 0.2152]` vs `probs_11=[0.2778, 0.2611, 0.2501, 0.2110]`（`diff = 0.3792`） |
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
        return 10.0 ** (self.snr_db / 10.0)


CONDITION_SPECS: Dict[str, ConditionSpec] = {
    "C1_coherent_2db": ConditionSpec(
        name="C1_coherent_2db",
        snr_db=2.0,
        h=0,
        ell=0,
        t_p=0.0,
        drift_type="coherent",
        in_support_encoder=True,
        in_support_connector=True,
    ),
    "C2_discrete_walk": ConditionSpec(
        name="C2_discrete_walk",
        snr_db=2.0,
        h=2,
        ell=2,
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


def dedup_references(references: Sequence[Reference]) -> Tuple[List[Reference], int, int, bool]:
    """Shared deduplication contract across direct readout and 3B host.

    Groups references by `(acquisition_id, shares_channel == True)` and verifies that
    identical acquisition_id entries do not point to conflicting samples.
    """
    seen_all: Dict[int, Reference] = {}
    unique_valid: List[Reference] = []
    raw_valid_count = 0
    for ref in references:
        samples = np.asarray(ref.samples, dtype=np.complex128)
        if ref.acquisition_id in seen_all:
            prev = seen_all[ref.acquisition_id]
            if bool(prev.shares_channel) != bool(ref.shares_channel) or abs(prev.t_p - ref.t_p) > 1e-12:
                raise ValueError(f"Conflicting metadata for acquisition_id={ref.acquisition_id}")
            if not np.array_equal(samples, np.asarray(prev.samples, dtype=np.complex128)):
                raise ValueError(f"Conflicting samples for acquisition_id={ref.acquisition_id}")
            if ref.shares_channel:
                raw_valid_count += 1
            continue
        seen_all[ref.acquisition_id] = ref
        if ref.shares_channel:
            raw_valid_count += 1
            unique_valid.append(ref)
    dedup_applied = raw_valid_count != len(unique_valid)
    return unique_valid, raw_valid_count, len(unique_valid), dedup_applied


def build_query_prompt(
    question_type: str,
    cond: ConditionSpec,
    protocol: str = "repeat_x1",
    ref_arm: str = "none",
    references: Sequence[Reference] = (),
    include_ref_token: bool = True,
    deduplicate: bool = True,
) -> Tuple[str, Tuple[str, ...]]:
    """Construct exact Chinese prompt (only public metadata + modality placeholder tokens).

    Placeholder tokens: `<W1>`, `<W2>` always present; `<REF>` present only when references
    are supplied (and if `deduplicate=False` for raw_duplicate_probe, emits one `<REF>` per
    raw reference entry so host-side sensitivity to duplicate tokens can be measured).
    """
    if question_type not in ("Q1", "Q2", "Q3", "Q3_after_pilot"):
        raise ValueError(f"Unknown question_type={question_type}")
    drift_desc_map = {
        "coherent": "相干静止(漂移=0)",
        "discrete": f"离散C4游走(d={cond.d:.2f}/�?",
        "continuous": f"连续高斯游走(sigma={cond.sigma_phi:.2f}rad/�?",
        "cfo_continuous": f"线性频�?游走(omega={cond.omega:.2f},sigma={cond.sigma_phi:.2f})",
    }
    meta = (
        f"工况={cond.name}; 信噪�?{cond.snr_db:.1f}dB; "
        f"两窗间隔h={cond.h}, 留出间隔ell={cond.ell}, 导频时刻tp={cond.t_p:.1f}; "
        f"信道={drift_desc_map[cond.drift_type]}。\n"
        f"已编码波形：<W1><W2>�?
    )
    if question_type == "Q1":
        return f"{meta}\n问题：{Q1_PROMPT}", Q1_CHOICES
    if question_type == "Q3":
        qtext = Q3_PROMPT_REPEAT if protocol == "repeat_x1" else Q3_PROMPT_INDEP
        return f"{meta}\n问题：{qtext}", Q3_CHOICES

    # Q2 or Q3_after_pilot (post reference reveal)
    if len(references) == 0:
        ref_text = "补充参考：无参考观测�?
    else:
        if deduplicate:
            seen_ids = set()
            items = []
            for r in references:
                if r.acquisition_id in seen_ids:
                    continue
                seen_ids.add(r.acquisition_id)
                tag = "同信�? if r.shares_channel else "独立无关"
                tok = "<REF>" if include_ref_token else ""
                items.append(f"[编号={r.acquisition_id},状�?{tag},时刻={r.t_p:.1f}]{tok}")
            dup_note = f"（已按唯一采集编号去重，共{len(items)}条独立参考）"
            ref_text = "补充参考：" + "�?.join(items) + dup_note + "�?
        else:
            items = []
            for r in references:
                tag = "同信�? if r.shares_channel else "独立无关"
                tok = "<REF>" if include_ref_token else ""
                items.append(f"[编号={r.acquisition_id},状�?{tag},时刻={r.t_p:.1f}]{tok}")
            ref_text = f"补充参考（未去重，共{len(items)}条）�? + "�?.join(items) + "�?

    qtext = Q2_PROMPT if question_type == "Q2" else Q3_PROMPT_REPEAT_AFTER_PILOT
    choices = Q2_CHOICES if question_type == "Q2" else Q3_CHOICES
    return f"{meta}\n{ref_text}\n问题：{qtext}", choices
research\abcde_convergence_2026_11_22\B\audit.py:135:                    np.testing.assert_allclose(out["B3_mixture_likelihood"]["llrs"][idx],rec["llr"],atol=3e-10)
research\realdata_ofdm_stress_and_theory\group_c_transport_attention.py:956:    b3_t_cal = float(gb["dev_calibration"]["llr_T_cal"]["B3_mixture_likelihood"])
research\abcde_convergence_2026_11_22\experiments\historical\real_smoke.py:72:        archived={(r['condition'],int(r['tti'])):float(r['ber']) for r in csv.DictReader(handle) if r['method']=='B3_mixture_likelihood'}
research\abcde_convergence_2026_11_22\experiments\historical\real_smoke.py:80:                                        ll_temp=TEMP,p_stay=STAY,use_hmm=True)['B3_mixture_likelihood']
research\realdata_ofdm_stress_and_theory\generate_figures.py:144:        ("B3_mixture_likelihood", "B3: Mix-Post LLR", "#1b7837"),
research\realdata_ofdm_stress_and_theory\generate_figures.py:176:    ax.plot(temps, [r["ber"]["B3_mixture_likelihood"] for r in amb], "o-", color="#1b7837", lw=2.2, label="B3 Mixture LLR BER")
research\realdata_ofdm_stress_and_theory\generate_figures.py:190:        f"Yet Mix-LLR BER={last['ber']['B3_mixture_likelihood']:.3f} << Lin={last['ber']['B2b_linear_fusion_totvar']:.3f}",
research\realdata_ofdm_stress_and_theory\generate_figures.py:218:            means = [gb["conditions"][c]["methods"]["B3_mixture_likelihood"]["ber"] for c in COND_ORDER]
research\realdata_ofdm_stress_and_theory\generate_figures.py:244:            means = [gb["conditions"][c]["methods"]["B3_mixture_likelihood"]["nll_cal"] for c in COND_ORDER]
research\realdata_ofdm_stress_and_theory\group_b_posterior_fusion.py:17:  - B3_mixture_likelihood    : FULL mixture posterior marginalized directly inside 16-QAM bit LLR (posterior_fusion §5.2!)
research\realdata_ofdm_stress_and_theory\group_b_posterior_fusion.py:69:    "B3_mixture_likelihood",
research\realdata_ofdm_stress_and_theory\group_b_posterior_fusion.py:139:        "B3_mixture_likelihood": mixture_16qam_llrs(
research\realdata_ofdm_stress_and_theory\group_b_posterior_fusion.py:153:        "B3_mixture_likelihood": h_mix,  # mixture decision uses full distribution; point-MSE benchmark = h_mix
research\realdata_ofdm_stress_and_theory\group_b_posterior_fusion.py:225:                    nlls.append(r["B3_mixture_likelihood"]["nll"])
research\realdata_ofdm_stress_and_theory\group_b_posterior_fusion.py:337:            ("B3_vs_B0", "B0_classical_single", "B3_mixture_likelihood"),
research\realdata_ofdm_stress_and_theory\group_b_posterior_fusion.py:338:            ("B3_vs_B0b", "B0b_classical_nbi_guard", "B3_mixture_likelihood"),
research\realdata_ofdm_stress_and_theory\group_b_posterior_fusion.py:339:            ("B3_vs_B1_hard", "B1_hard_select", "B3_mixture_likelihood"),
research\realdata_ofdm_stress_and_theory\group_b_posterior_fusion.py:340:            ("B3_vs_B2a_lin", "B2a_linear_fusion_within", "B3_mixture_likelihood"),
research\realdata_ofdm_stress_and_theory\group_b_posterior_fusion.py:341:            ("B3_vs_B2b_totvar", "B2b_linear_fusion_totvar", "B3_mixture_likelihood"),
research\realdata_ofdm_stress_and_theory\group_b_posterior_fusion.py:393:        m_ber = {m: [] for m in ("B1_hard_select", "B2a_linear_fusion_within", "B2b_linear_fusion_totvar", "B3_mixture_likelihood")}
research\realdata_ofdm_stress_and_theory\group_b_posterior_fusion.py:395:        m_nll = {m: [] for m in ("B1_hard_select", "B2a_linear_fusion_within", "B2b_linear_fusion_totvar", "B3_mixture_likelihood")}
research\realdata_ofdm_stress_and_theory\group_b_posterior_fusion.py:454:            f"lin_tv={row['ber']['B2b_linear_fusion_totvar']:.4f}, MIX_LLR={row['ber']['B3_mixture_likelihood']:.4f}"
```

### Event #14: `item.completed` — `command_execution` (`item_11`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n 'y.*sqrt|complex_noise|snr|def.*sample' research/matched_cache -g '*.py'; rg -n 'def.*cache|detach|clone|use_cache' research/stage3d_qwen_signed_phase -g '*.py'; git ls-tree -r --name-only origin/research/realdata-cttc-indoor-qpsk research/realdata_cttc_qpsk | Select-String -Pattern 'anchor|manifest|split|config' ; Get-ChildItem -Force research/stage4_math_and_design"
```

**Aggregated Output:**

```text
research/matched_cache\audit.py:99:    means = np.asarray([1, 1j, -1, -1j]) / np.sqrt(3)
research/stage3d_qwen_signed_phase\qwen_connector.py:192:def _make_differentiable_qwen35_cache(lm_config):
research/stage3d_qwen_signed_phase\qwen_connector.py:205:def _clone_expand_qwen35_cache(pref_cache, lm_config, repeats: int):
research/stage3d_qwen_signed_phase\qwen_connector.py:346:        embed_weight = self.lm.get_input_embeddings().weight.detach().float()
research/stage3d_qwen_signed_phase\qwen_connector.py:509:    def score_candidates_naive_uncached(
research/stage3d_qwen_signed_phase\qwen_connector.py:528:            fwd_kwargs = dict(inputs_embeds=full_emb, attention_mask=attn, position_ids=pos, use_cache=False)
research/stage3d_qwen_signed_phase\qwen_connector.py:537:    def score_candidates_batched_uncached(
research/stage3d_qwen_signed_phase\qwen_connector.py:570:        fwd_kwargs = dict(inputs_embeds=full_emb, attention_mask=full_attn, position_ids=pos, use_cache=False)
research/stage3d_qwen_signed_phase\qwen_connector.py:606:            use_cache=(max_len > 1),
research/stage3d_qwen_signed_phase\qwen_connector.py:637:            exp_cache = _clone_expand_qwen35_cache(out_cache, self.lm.config, repeats=4)
research/stage3d_qwen_signed_phase\qwen_connector.py:656:                use_cache=False,
research/stage3d_qwen_signed_phase\modal_runner.py:143:def remote_eval_dev_pilot(batch_size: int = 64, scorer_mode: str = "kv_cached") -> bytes:
research/stage3d_qwen_signed_phase\modal_runner.py:171:def remote_eval_blind(batch_size: int = 64, scorer_mode: str = "kv_cached") -> dict:
research/stage3d_qwen_signed_phase\modal_runner.py:316:                chunks.append(lp.exp().detach().cpu().numpy().astype(np.float64))
research/stage3d_qwen_signed_phase\modal_runner.py:461:                gnorm = sum(float(p.grad.detach().float().norm().item()) for p in conn.parameters() if p.grad is not None)
research/stage3d_qwen_signed_phase\modal_runner.py:541:            dir_1_list.append(d1.detach().cpu().numpy().astype(np.float64)[0])
research/stage3d_qwen_signed_phase\modal_runner.py:542:        dir_64_np = dir_64.detach().cpu().numpy().astype(np.float64)
research/stage3d_qwen_signed_phase\modal_runner.py:543:        dir_16_np = dir_16.detach().cpu().numpy().astype(np.float64)
research/stage3d_qwen_signed_phase\train_eval_stage3d.py:314:    loss = loss_q3.detach() + loss_q2.detach()
research/stage3d_qwen_signed_phase\train_eval_stage3d.py:319:        n: float(p.grad.detach().float().norm().item()) if p.grad is not None else 0.0
research/stage3d_qwen_signed_phase\train_eval_stage3d.py:520:            lp_parts.append(lp_u.detach().cpu().numpy().astype(np.float64))
research/stage3d_qwen_signed_phase\train_eval_stage3d.py:521:            raw_parts.append(raw_u.detach().cpu().numpy().astype(np.float64))
research/stage3d_qwen_signed_phase\train_eval_stage3d.py:729:            best_state = {k: v.detach().cpu().clone() for k, v in connector.state_dict().items()}
research/stage3d_qwen_signed_phase\train_eval_stage3d.py:766:                        arr_ll = raw_ll.detach().cpu().numpy().astype(np.float64)
research/stage3d_qwen_signed_phase\train_eval_stage3d.py:837:def score_256_cache_table(
research/stage3d_qwen_signed_phase\train_eval_stage3d.py:872:            chunks.append(raw_ll.detach().cpu().numpy().astype(np.float64))
research/stage3d_qwen_signed_phase\train_eval_stage3d.py:1013:            direct_np = direct_ll.detach().cpu().numpy().astype(np.float64)
research/stage3d_qwen_signed_phase\train_eval_stage3d.py:1034:                d16_np = dir_16.detach().cpu().numpy().astype(np.float64)
research/stage3d_qwen_signed_phase\train_eval_stage3d.py:1095:                        chunks.append(raw_ll.detach().cpu().numpy().astype(np.float64))
research/stage3d_qwen_signed_phase\train_eval_stage3d.py:1131:                            chunks.append(raw_ll.detach().cpu().numpy().astype(np.float64))

research/realdata_cttc_qpsk/data_manifest.json
research/realdata_cttc_qpsk/results/capture_anchors.json

PSPath            : Microsoft.PowerShell.Core\FileSystem::C:\Users\������\.gemini\antigravity\scratch\EM-FloatFormer-st
                    age4\research\stage4_math_and_design\discussions
PSParentPath      : Microsoft.PowerShell.Core\FileSystem::C:\Users\������\.gemini\antigravity\scratch\EM-FloatFormer-st
                    age4\research\stage4_math_and_design
PSChildName       : discussions
PSDrive           : C
PSProvider        : Microsoft.PowerShell.Core\FileSystem
PSIsContainer     : True
Name              : discussions
FullName          : C:\Users\������\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\d
                    iscussions
Parent            : stage4_math_and_design
Exists            : True
Root              : C:\
Extension         : 
CreationTime      : 2026/11/26 19:11:33
CreationTimeUtc   : 2026/11/26 19:11:33
LastAccessTime    : 2026/11/26 19:15:31
LastAccessTimeUtc : 2026/11/26 19:15:31
LastWriteTime     : 2026/11/26 19:11:33
LastWriteTimeUtc  : 2026/11/26 19:11:33
Attributes        : Directory
Mode              : d-----
BaseName          : discussions
Target            : {}
LinkType          : 


PSPath            : Microsoft.PowerShell.Core\FileSystem::C:\Users\������\.gemini\antigravity\scratch\EM-FloatFormer-st
                    age4\research\stage4_math_and_design\M1
PSParentPath      : Microsoft.PowerShell.Core\FileSystem::C:\Users\������\.gemini\antigravity\scratch\EM-FloatFormer-st
                    age4\research\stage4_math_and_design
PSChildName       : M1
PSDrive           : C
PSProvider        : Microsoft.PowerShell.Core\FileSystem
PSIsContainer     : True
Name              : M1
FullName          : C:\Users\������\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M
                    1
Parent            : stage4_math_and_design
Exists            : True
Root              : C:\
Extension         : 
CreationTime      : 2026/11/26 19:11:33
CreationTimeUtc   : 2026/11/26 19:11:33
LastAccessTime    : 2026/11/26 19:15:31
LastAccessTimeUtc : 2026/11/26 19:15:31
LastWriteTime     : 2026/11/26 19:11:33
LastWriteTimeUtc  : 2026/11/26 19:11:33
Attributes        : Directory
Mode              : d-----
BaseName          : M1
Target            : {}
LinkType          : 


PSPath            : Microsoft.PowerShell.Core\FileSystem::C:\Users\������\.gemini\antigravity\scratch\EM-FloatFormer-st
                    age4\research\stage4_math_and_design\M2
PSParentPath      : Microsoft.PowerShell.Core\FileSystem::C:\Users\������\.gemini\antigravity\scratch\EM-FloatFormer-st
                    age4\research\stage4_math_and_design
PSChildName       : M2
PSDrive           : C
PSProvider        : Microsoft.PowerShell.Core\FileSystem
PSIsContainer     : True
Name              : M2
FullName          : C:\Users\������\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M
                    2
Parent            : stage4_math_and_design
Exists            : True
Root              : C:\
Extension         : 
CreationTime      : 2026/11/26 19:11:33
CreationTimeUtc   : 2026/11/26 19:11:33
LastAccessTime    : 2026/11/26 19:15:31
LastAccessTimeUtc : 2026/11/26 19:15:31
LastWriteTime     : 2026/11/26 19:11:33
LastWriteTimeUtc  : 2026/11/26 19:11:33
Attributes        : Directory
Mode              : d-----
BaseName          : M2
Target            : {}
LinkType          : 


PSPath            : Microsoft.PowerShell.Core\FileSystem::C:\Users\������\.gemini\antigravity\scratch\EM-FloatFormer-st
                    age4\research\stage4_math_and_design\M3
PSParentPath      : Microsoft.PowerShell.Core\FileSystem::C:\Users\������\.gemini\antigravity\scratch\EM-FloatFormer-st
                    age4\research\stage4_math_and_design
PSChildName       : M3
PSDrive           : C
PSProvider        : Microsoft.PowerShell.Core\FileSystem
PSIsContainer     : True
Name              : M3
FullName          : C:\Users\������\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M
                    3
Parent            : stage4_math_and_design
Exists            : True
Root              : C:\
Extension         : 
CreationTime      : 2026/11/26 19:11:33
CreationTimeUtc   : 2026/11/26 19:11:33
LastAccessTime    : 2026/11/26 19:15:31
LastAccessTimeUtc : 2026/11/26 19:15:31
LastWriteTime     : 2026/11/26 19:11:33
LastWriteTimeUtc  : 2026/11/26 19:11:33
Attributes        : Directory
Mode              : d-----
BaseName          : M3
Target            : {}
LinkType          : 


PSPath            : Microsoft.PowerShell.Core\FileSystem::C:\Users\������\.gemini\antigravity\scratch\EM-FloatFormer-st
                    age4\research\stage4_math_and_design\M4
PSParentPath      : Microsoft.PowerShell.Core\FileSystem::C:\Users\������\.gemini\antigravity\scratch\EM-FloatFormer-st
                    age4\research\stage4_math_and_design
PSChildName       : M4
PSDrive           : C
PSProvider        : Microsoft.PowerShell.Core\FileSystem
PSIsContainer     : True
Name              : M4
FullName          : C:\Users\������\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M
                    4
Parent            : stage4_math_and_design
Exists            : True
Root              : C:\
Extension         : 
CreationTime      : 2026/11/26 19:11:33
CreationTimeUtc   : 2026/11/26 19:11:33
LastAccessTime    : 2026/11/26 19:15:31
LastAccessTimeUtc : 2026/11/26 19:15:31
LastWriteTime     : 2026/11/26 19:11:33
LastWriteTimeUtc  : 2026/11/26 19:11:33
Attributes        : Directory
Mode              : d-----
BaseName          : M4
Target            : {}
LinkType          : 


PSPath            : Microsoft.PowerShell.Core\FileSystem::C:\Users\������\.gemini\antigravity\scratch\EM-FloatFormer-st
                    age4\research\stage4_math_and_design\__pycache__
PSParentPath      : Microsoft.PowerShell.Core\FileSystem::C:\Users\������\.gemini\antigravity\scratch\EM-FloatFormer-st
                    age4\research\stage4_math_and_design
PSChildName       : __pycache__
PSDrive           : C
PSProvider        : Microsoft.PowerShell.Core\FileSystem
PSIsContainer     : True
Name              : __pycache__
FullName          : C:\Users\������\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\_
                    _pycache__
Parent            : stage4_math_and_design
Exists            : True
Root              : C:\
Extension         : 
CreationTime      : 2026/11/26 19:13:09
CreationTimeUtc   : 2026/11/26 19:13:09
LastAccessTime    : 2026/11/26 19:15:31
LastAccessTimeUtc : 2026/11/26 19:15:31
LastWriteTime     : 2026/11/26 19:13:09
LastWriteTimeUtc  : 2026/11/26 19:13:09
Attributes        : Directory
Mode              : d-----
BaseName          : __pycache__
Target            : {}
LinkType          : 


LastWriteTime : 2026/11/26 19:13:07
Length        : 8800
Name          : transcript_utils.py



```

### Event #15: `item.completed` — `command_execution` (`item_12`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command 'git show origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/capture_anchors.json'
```

**Aggregated Output:**

```text
[
  {
    "capture_id": "f2450_snr20_QPSK",
    "freq_mhz": 2450,
    "nom_snr_db": 20,
    "timestamp": "Wed May 19 17:05:16 2021",
    "n_samples": 22001908,
    "bg_noise_var": 3.0642262061064685e-09,
    "sig_total_pow": 5.753840639499685e-07,
    "measured_snr_db": 22.71317977218708,
    "anchor_rx_pos": 4000000.717744379,
    "anchor_src_sym": 492892,
    "sps_eff": 4.000002029292535,
    "sfo_ppm": 0.507323133724924,
    "offline_cfo_med_hz": -1230.2398681640625,
    "offline_cfo_min_hz": -1258.85009765625,
    "offline_cfo_max_hz": -1213.07373046875,
    "secondary_spur_ratio": 0.02665338284959226,
    "secondary_spur_offset_hz": 72.479248046875,
    "bit_alignment": 0,
    "lock_norm_corr_mean": 0.944880921028427,
    "stream_source": "file_2.dat"
  },
  {
    "capture_id": "f2450_snr10_QPSK",
    "freq_mhz": 2450,
    "nom_snr_db": 10,
    "timestamp": "Wed May 19 17:01:52 2021",
    "n_samples": 22001908,
    "bg_noise_var": 3.0842834952693465e-09,
    "sig_total_pow": 3.894835387541207e-08,
    "measured_snr_db": 10.655052882611155,
    "anchor_rx_pos": 4000003.273988869,
    "anchor_src_sym": 492871,
    "sps_eff": 4.000001923272496,
    "sfo_ppm": 0.4808181239823739,
    "offline_cfo_med_hz": -1173.0194091796875,
    "offline_cfo_min_hz": -1194.000244140625,
    "offline_cfo_max_hz": -1161.5753173828125,
    "secondary_spur_ratio": 0.026056981047085023,
    "secondary_spur_offset_hz": 72.479248046875,
    "bit_alignment": 0,
    "lock_norm_corr_mean": 0.9427839545077384,
    "stream_source": "file_2.dat"
  },
  {
    "capture_id": "f2450_snr0_QPSK",
    "freq_mhz": 2450,
    "nom_snr_db": 0,
    "timestamp": "Wed May 19 16:53:51 2021",
    "n_samples": 22001908,
    "bg_noise_var": 2.9808777668449693e-09,
    "sig_total_pow": 8.304713805529218e-09,
    "measured_snr_db": 2.518805039625829,
    "anchor_rx_pos": 4000001.4816942867,
    "anchor_src_sym": 493763,
    "sps_eff": 4.000002092946086,
    "sfo_ppm": 0.523236521443593,
    "offline_cfo_med_hz": -1283.6456298828125,
    "offline_cfo_min_hz": -1296.9970703125,
    "offline_cfo_max_hz": -1251.220703125,
    "secondary_spur_ratio": 0.02642627121178752,
    "secondary_spur_offset_hz": 70.5718994140625,
    "bit_alignment": 0,
    "lock_norm_corr_mean": 0.7681396946252546,
    "stream_source": "file_2.dat"
  },
  {
    "capture_id": "f708_snr20_QPSK",
    "freq_mhz": 708,
    "nom_snr_db": 20,
    "timestamp": "Wed May 19 15:52:45 2021",
    "n_samples": 22001908,
    "bg_noise_var": 2.7648849876271697e-09,
    "sig_total_pow": 6.994473267241119e-07,
    "measured_snr_db": 24.013577948475305,
    "anchor_rx_pos": 4000001.7397749024,
    "anchor_src_sym": 493601,
    "sps_eff": 4.000002218034584,
    "sfo_ppm": 0.5545086458891291,
    "offline_cfo_med_hz": -391.0064697265625,
    "offline_cfo_min_hz": -396.728515625,
    "offline_cfo_max_hz": -381.4697265625,
    "secondary_spur_ratio": 0.031379307089864035,
    "secondary_spur_offset_hz": 1586.9140625,
    "bit_alignment": 0,
    "lock_norm_corr_mean": 0.9441099091373244,
    "stream_source": "file_2.dat"
  },
  {
    "capture_id": "f708_snr10_QPSK",
    "freq_mhz": 708,
    "nom_snr_db": 10,
    "timestamp": "Wed May 19 16:14:10 2021",
    "n_samples": 22001908,
    "bg_noise_var": 2.777065688519542e-09,
    "sig_total_pow": 4.132570907700028e-08,
    "measured_snr_db": 11.424229462348215,
    "anchor_rx_pos": 4000000.456007515,
    "anchor_src_sym": 493414,
    "sps_eff": 4.0000021036173665,
    "sfo_ppm": 0.5259043416305786,
    "offline_cfo_med_hz": -373.84033203125,
    "offline_cfo_min_hz": -387.1917724609375,
    "offline_cfo_max_hz": -364.3035888671875,
    "secondary_spur_ratio": 0.2604101834998875,
    "secondary_spur_offset_hz": 1502.99072265625,
    "bit_alignment": 0,
    "lock_norm_corr_mean": 0.9173542840129798,
    "stream_source": "file_2.dat"
  },
  {
    "capture_id": "f708_snr0_QPSK",
    "freq_mhz": 708,
    "nom_snr_db": 0,
    "timestamp": "Wed May 19 16:22:27 2021",
    "n_samples": 22001908,
    "bg_noise_var": 2.8434006260624756e-09,
    "sig_total_pow": 4.1981009779590295e-09,
    "measured_snr_db": -3.219948110894876,
    "anchor_rx_pos": 4000000.960871691,
    "anchor_src_sym": 492771,
    "sps_eff": 4.00000209393422,
    "sfo_ppm": 0.5234835549483563,
    "offline_cfo_med_hz": -371.9329833984375,
    "offline_cfo_min_hz": -377.655029296875,
    "offline_cfo_max_hz": -368.1182861328125,
    "secondary_spur_ratio": 0.04297073540440458,
    "secondary_spur_offset_hz": -20164.48974609375,
    "bit_alignment": 0,
    "lock_norm_corr_mean": 0.5561941469062333,
    "stream_source": "file_2.dat"
  },
  {
    "capture_id": "f433_snr20_QPSK",
    "freq_mhz": 433,
    "nom_snr_db": 20,
    "timestamp": "Wed May 19 16:46:19 2021",
    "n_samples": 22001908,
    "bg_noise_var": 3.1134732569881862e-09,
    "sig_total_pow": 5.563989304846473e-07,
    "measured_snr_db": 22.49704115473132,
    "anchor_rx_pos": 4000002.3574077785,
    "anchor_src_sym": 235640,
    "sps_eff": 4.000001561883662,
    "sfo_ppm": 0.39047091560107106,
    "offline_cfo_med_hz": -167.8466796875,
    "offline_cfo_min_hz": -175.47607421875,
    "offline_cfo_max_hz": -165.9393310546875,
    "secondary_spur_ratio": 0.028704732473488415,
    "secondary_spur_offset_hz": -72.479248046875,
    "bit_alignment": 1,
    "lock_norm_corr_mean": 0.9812004739315068,
    "stream_source": "file_2.dat"
  },
  {
    "capture_id": "f433_snr10_QPSK",
    "freq_mhz": 433,
    "nom_snr_db": 10,
    "timestamp": "Wed May 19 16:42:45 2021",
    "n_samples": 22001908,
    "bg_noise_var": 3.0300049136400276e-09,
    "sig_total_pow": 4.8832287546929365e-08,
    "measured_snr_db": 11.79443789553224,
    "anchor_rx_pos": 4000003.2122689085,
    "anchor_src_sym": 492790,
    "sps_eff": 4.000002020701151,
    "sfo_ppm": 0.5051752878237892,
    "offline_cfo_med_hz": -219.3450927734375,
    "offline_cfo_min_hz": -225.067138671875,
    "offline_cfo_max_hz": -213.623046875,
    "secondary_spur_ratio": 0.028669206528786382,
    "secondary_spur_offset_hz": -74.3865966796875,
    "bit_alignment": 0,
    "lock_norm_corr_mean": 0.9569894697397523,
    "stream_source": "file_2.dat"
  },
  {
    "capture_id": "f433_snr0_QPSK",
    "freq_mhz": 433,
    "nom_snr_db": 0,
    "timestamp": "Wed May 19 16:38:12 2021",
    "n_samples": 22001908,
    "bg_noise_var": 3.101694234786123e-09,
    "sig_total_pow": 1.1637696584188006e-08,
    "measured_snr_db": 4.3965554313904,
    "anchor_rx_pos": 4000003.3067990313,
    "anchor_src_sym": 235845,
    "sps_eff": 4.000001454072232,
    "sfo_ppm": 0.3635180580996433,
    "offline_cfo_med_hz": -158.3099365234375,
    "offline_cfo_min_hz": -160.21728515625,
    "offline_cfo_max_hz": -156.402587890625,
    "secondary_spur_ratio": 0.029393269517622667,
    "secondary_spur_offset_hz": -74.3865966796875,
    "bit_alignment": 1,
    "lock_norm_corr_mean": 0.8485045389613812,
    "stream_source": "file_2.dat"
  }
]
```

### Event #16: `item.completed` — `command_execution` (`item_13`)

- Status: `failed`, Exit Code: `1`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "git ls-tree -r --name-only origin/research/realdata-pluto-qpsk research/realdata_pluto_qpsk | Select-String -Pattern 'manifest|result'; rg -n '"'^(def|class)|SNR|sigma'"' research/matched_cache/codec.py research/matched_cache/physics.py; Get-ChildItem -Force -Name research/stage4_math_and_design/M4; rg -n 'uv|python' .gitignore"
```

**Aggregated Output:**

```text

research/realdata_pluto_qpsk/metadata/data_manifest.json
research/realdata_pluto_qpsk/results/ab_paired_stats.json
research/realdata_pluto_qpsk/results/ab_test_summary.json
research/realdata_pluto_qpsk/results/benchmark_prep_summary.json
research/realdata_pluto_qpsk/results/c_neural_test_summary.json
research/realdata_pluto_qpsk/results/dev_tuning_summary.json
research/realdata_pluto_qpsk/results/frontend_64dev_verify.json
research/realdata_pluto_qpsk/results/test_per_frame_summary.csv.gz
rg: research/matched_cache/codec.py: 系统找不到指定的文件。 (os error 2)
rg: research/matched_cache/physics.py: 系统找不到指定的文件。 (os error 2)


```

### Event #17: `item.completed` — `command_execution` (`item_14`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command 'git show origin/research/realdata-pluto-qpsk:research/realdata_pluto_qpsk/metadata/data_manifest.json'
```

**Aggregated Output:**

```text
{
  "dataset_title": "Frames of SDR ADALM PLUTO based QPSK Communication Link ",
  "dataset_doi": "10.17632/k72b4pnc5g.1",
  "dataset_url": "https://data.mendeley.com/datasets/k72b4pnc5g/1",
  "public_api_metadata_url": "https://data.mendeley.com/public-api/datasets/k72b4pnc5g?version=1",
  "public_api_files_url": "https://data.mendeley.com/public-api/datasets/k72b4pnc5g/files?folder_id=root&version=1",
  "download_url": "https://data.mendeley.com/public-files/datasets/k72b4pnc5g/files/b494a472-ab2e-4d39-99ca-86866b76f159/file_downloaded",
  "download_method": "urllib_stream",
  "download_elapsed_sec": 10.073,
  "filename": "QPSKlink.mat",
  "size_bytes": 822762822,
  "sha256": "c029532908f9702fea4be3ce37d0e1de9138019e12dd06e228abb1b83525f2b7",
  "sha256_matches_mendeley_api": true,
  "is_hdf5_v73": false,
  "header_preview": "MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: Tue Jun  2 15:58:38 2026",
  "metadata_sha256": "efdb210b2b79f795effc4c583bab981b4a452dd018b365d0be2e976dafdc426a",
  "files_metadata_sha256": "f09342aeb2953d283651ab595a1e11cbce85606d4f5affd4eff3841b111ba02d",
  "verified_utc": "2026-11-21T22:43:49Z"
}
```

### Event #18: `item.completed` — `command_execution` (`item_15`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "Get-Content -Encoding utf8 research/stage3d_qwen_signed_phase/REPORT.md | Select-Object -Skip 96 -First 16; rg -n 'def|noisy|sqrt\\(' research/matched_cache/generate.py; rg --files research/matched_cache -g '*.py'; Get-ChildItem -Name research/stage4_math_and_design/discussions"
```

**Aggregated Output:**

```text
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

rg: research/matched_cache/generate.py: IO error for operation on research/matched_cache/generate.py: 系统找不到指定的文件。 (os error 2)
research/matched_cache\learned.py
research/matched_cache\audit.py
research/matched_cache\compare.py
research/matched_cache\evaluate.py
research/matched_cache\train.py
research/matched_cache\physical.py
M1
M2
M3
M4
```

### Event #19: `item.completed` — `agent_message` (`item_16`)

The branch code exposes two important protocol gaps. Pluto’s legacy CFO estimator reads across the capture, and CTTC’s legacy alignment uses known source symbols at several capture times. I’ll keep both as historical references and specify a causal frontend for the Stage 4 comparison. I’ll also test the repeated scalar embedding and its full-rank counterpart on the frozen encoder.

### Event #20: `item.completed` — `command_execution` (`item_18`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n 'matched|independent|limitations|replay' 'C:/Users/李则徐/.codex/memories/rollout_summaries/2026-11-17T20-54-26-dFGP-p4_em_native_transformer_assessment_and_github_publication.md'"
```

**Aggregated Output:**

```text
22:- P4-B original benchmark: 23 condition/SNR cells, 7,176 coded blocks per receiver. Original EMFloatFormer2D errored on 4,476 blocks; fixed-prior 2D-LMMSE on 4,307; matched-statistics LMMSE on 2,548.
24:- Follow-up receiver experiments reduced errors on unseen delay distributions, but the pure Fourier/逐径 pilot front-end alone achieved nearly all of the gain. Therefore do not attribute this improvement to Transformer attention without matched-front-end controls.
32:- Completed independent replay of 150 coded rows, 10 final model checkpoints, 80 one-dimensional factorial arms, and 240 gain-conditioned observations.
36:- Formal two-dimensional result: EM did not significantly outperform fixed-prior classical baselines in any primary comparison. In-distribution at 10 dB/strong PA, EM had 3,266/4,160 block errors versus fixed LMMSE 3,227 and fixed IDD-PA 3,211; EM was significantly worse than fixed IDD-PA after Holm correction (`p=0.0234375`). Long-delay EM decoded only 22/4,160 blocks successfully, though matched-statistics baselines were much better.
48:- Re-cloned GitHub and verified the public replay entrypoint, 150 coded rows, ten weights, both Holm families, and both test receipts.
65:Conclusion: P4 made substantial algorithmic progress by implementing and testing a complex-valued, gauge-equivariant, Hermitian-attention Transformer prototype with physical channel simulation and coded-link evaluation. The strongest evidence is that signal symmetries and complex-valued operations were encoded directly into the model and some mechanisms showed reproducible value in controlled 1D ablations. However, the formal 2D result did not establish a learned-Transformer advantage; the largest 2D improvements came from better pilot/channel front-ends. The next decisive experiment is a strictly matched comparison where EM and classical receivers share the same physical front-end, pilot information, decoder, and inference budget, isolating only the learned module.
```

### Event #21: `item.completed` — `command_execution` (`item_17`)

- Status: `failed`, Exit Code: `1`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "uv cache dir; uv python find 3.12; Get-ChildItem Env:UV*; rg -n 'unit|noise|sample_window|draw' research/matched_cache/train.py research/matched_cache/physical.py"
```

**Aggregated Output:**

```text
C:\Users\李则徐\AppData\Local\uv\cache
error: Failed to initialize cache at `C:\Users\李则徐\AppData\Local\uv\cache`
  Caused by: failed to open file `C:\Users\李则徐\AppData\Local\uv\cache\sdists-v9\.git`: 拒绝访问。 (os error 5)
```

### Event #22: `item.completed` — `command_execution` (`item_20`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "Get-ChildItem -Force -Name; Get-ChildItem Env: | Where-Object Name -Match 'UV|PIP|PYTHON'; Get-ChildItem 'C:/Users/李则徐/AppData/Local/uv/cache/archive-v0' -Name | Select-Object -First 8; Get-ChildItem Env:TEMP"
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

Name                           Value                                                                                   
----                           -----                                                                                   
PIP_DISABLE_PIP_VERSION_CHECK  1                                                                                       
PIP_NO_INDEX                   1                                                                                       
PYTHONIOENCODING               utf-8                                                                                   
PYTHONUTF8                     1                                                                                       
-3Eg0_w5zGyPhAVD
-INXTsup9SBW2nnw
-yFVWlDN8f_PSvOm
0kEkF80-N13tUw4M
0xUSRPAHLRdjp9oW
12oJCSOQV4GMwgHu
1SmvAy6YA_w5SdKx
1uFPYbapY1T1Qpz9
TEMP                           C:\WINDOWS\TEMP                                                                         


```

### Event #23: `item.completed` — `command_execution` (`item_19`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "Get-Command uv; Get-ChildItem 'C:/Users/李则徐/AppData/Local/uv/cache' -Name; Get-ChildItem 'C:/Users/李则徐/AppData/Roaming/uv/python' -Name; Get-ChildItem 'C:/Users/李则徐/AppData/Local/Programs/Python' -Name; rg -n 'def |noise|sqrt' research/matched_cache/train.py research/matched_cache/physical.py"
```

**Aggregated Output:**

```text

CommandType     Name                                               Version    Source                                   
-----------     ----                                               -------    ------                                   
Application     uv.exe                                             0.0.0.0    C:\qazwsx\Scripts\uv.exe                 
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
Get-ChildItem : �Ҳ���·����C:\Users\������\AppData\Roaming\uv\python������Ϊ��·�������ڡ�
����λ�� ��:2 �ַ�: 76
+ ... che' -Name; Get-ChildItem 'C:/Users/������/AppData/Roaming/uv/python' -N ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : ObjectNotFound: (C:\Users\������\AppData\Roaming\uv\python:String) [Get-ChildItem], ItemN
ot    FoundException
    + FullyQualifiedErrorId : PathNotFound,Microsoft.PowerShell.Commands.GetChildItemCommand
 
Python313
research/matched_cache/train.py:20:def training_data():
research/matched_cache/train.py:29:def name(variant, bits, seed):
research/matched_cache/train.py:33:def train_one(setting):
research/matched_cache/train.py:45:def main():
research/matched_cache/physical.py:11:MEAN = 1 / math.sqrt(2)
research/matched_cache/physical.py:12:NOISE_SD = 1 / math.sqrt(2 * SNR)
research/matched_cache/physical.py:18:def sign_bins(count):
research/matched_cache/physical.py:37:    def __init__(self, bits):
research/matched_cache/physical.py:50:    def encode(self, windows):
research/matched_cache/physical.py:54:        t = data.sum(axis=1) / math.sqrt(WINDOW_LENGTH)
research/matched_cache/physical.py:56:        real = (t.real - t.imag) / math.sqrt(2)
research/matched_cache/physical.py:57:        imag = (t.real + t.imag) / math.sqrt(2)
research/matched_cache/physical.py:63:def self_check():
research/matched_cache/physical.py:79:        def density(t):
research/matched_cache/physical.py:80:            norm = 1 / (2 * NOISE_SD * math.sqrt(2 * math.pi))


```

### Event #24: `item.completed` — `command_execution` (`item_21`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "Get-ChildItem 'C:/Users/李则徐/AppData/Local/uv' -Name; Get-ChildItem 'C:/qazwsx' -Name; where.exe python; where.exe py; rg --files --hidden 'C:/Users/李则徐/AppData/Local/uv/cache/wheels-v6' | rg 'numpy.*2\\.3\\.5|scipy.*1\\.17\\.0'; Get-Content research/matched_cache/train.py -TotalCount 34"
```

**Aggregated Output:**

```text
cache
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
C:\qazwsx\python.exe
C:\114514\python.exe
C:\376381\miniconda3\python.exe
C:\Users\������\AppData\Local\Microsoft\WindowsApps\python.exe
C:\Windows\py.exe
C:/Users/?????AppData/Local/uv/cache/wheels-v6\pypi\numpy\numpy-2.3.5-cp312-cp312-win_amd64.lock
C:/Users/?????AppData/Local/uv/cache/wheels-v6\pypi\numpy\2.3.5-cp312-cp312-win_amd64.msgpack
C:/Users/?????AppData/Local/uv/cache/wheels-v6\pypi\numpy\2.3.5-cp312-cp312-win_amd64.http
C:/Users/?????AppData/Local/uv/cache/wheels-v6\pypi\numpy\2.3.5-cp312-cp312-win_amd64
C:/Users/?????AppData/Local/uv/cache/wheels-v6\pypi\scipy\1.17.0-cp312-cp312-win_amd64.http
C:/Users/?????AppData/Local/uv/cache/wheels-v6\pypi\scipy\1.17.0-cp312-cp312-win_amd64
C:/Users/?????AppData/Local/uv/cache/wheels-v6\pypi\scipy\1.17.0-cp312-cp312-win_amd64.msgpack
C:/Users/?????AppData/Local/uv/cache/wheels-v6\pypi\scipy\scipy-1.17.0-cp312-cp312-win_amd64.lock
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
```

### Event #25: `item.completed` — `command_execution` (`item_22`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "py -0p; Get-ChildItem 'C:/Users/李则徐/AppData/Local/uv/cache/wheels-v6/pypi/numpy' -Force | Where-Object Name -Like '2.3.5-cp312*' | Format-List Name,LinkType,Target; Get-ChildItem 'C:/Users/李则徐/AppData/Local/uv/cache/environments-v2' -Name; rg -n 'app.function|cpu=|memory=|gpu=' research/stage3d_qwen_signed_phase/modal_runner.py; rg -n 'def |SNR|NOISE' research/fourphase_paired_questions/paired.py"
```

**Aggregated Output:**

```text
 -V:3.13 *        C:\Users\李则徐\AppData\Local\Programs\Python\Python313\python.exe


Name     : 2.3.5-cp312-cp312-win_amd64
LinkType : 
Target   : {}

Name     : 2.3.5-cp312-cp312-win_amd64.http
LinkType : 
Target   : {}

Name     : 2.3.5-cp312-cp312-win_amd64.msgpack
LinkType : 
Target   : {}



1093dab72d22fcbe
69:@app.function(
71:    gpu="A100-80GB",
86:@app.function(
88:    gpu="A100-80GB",
137:@app.function(
139:    gpu="A100-80GB",
165:@app.function(
167:    gpu="A100-80GB",
201:@app.function(
203:    gpu="A100-80GB",
354:@app.function(
356:    gpu="A100-80GB",
15:SNR = 10 ** 0.2  # E_s/N_0 = 2 dB, signal energy per window is one
30:def rotate(window, quarter_turns):
34:def parents(split):
37:    noise_sd = math.sqrt(1 / (2 * SNR))
39:    def observe(label):
55:def reference_views(pilot, other):
61:def features(win, view):
75:def likelihood(win):
77:    logw = [2 * SNR / math.sqrt(N) * (total * s.conjugate()).real for s in ROOTS]
81:def normalize_exp(logw):
87:def diff_prob(left, right):
92:def cache(data, view):
97:def read(cache_value, references, view):
103:    logw = [sum(2 * SNR / math.sqrt(N) * (sum(w) * s.conjugate()).real
108:def shift(probs, steps):
112:def maxdelta(a, b):
116:def argmax(p):
120:def iq_pairs(window):
124:def cases(split):
141:def run(split):
```

### Event #26: `item.completed` — `command_execution` (`item_23`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "Get-ChildItem 'C:/Users/李则徐/AppData/Local/uv/cache/wheels-v6/pypi/numpy/2.3.5-cp312-cp312-win_amd64' -Name; Get-ChildItem 'C:/Users/李则徐/AppData/Local/uv/cache/environments-v2/1093dab72d22fcbe' -Force -Recurse -Filter python.exe | Select-Object -ExpandProperty FullName; Get-ChildItem 'C:/Users/李则徐/.gemini/antigravity/scratch' -Name | Select-Object -First 25; Get-ChildItem 'C:/114514' -Name | Select-Object -First 12; Get-ChildItem 'C:/Users/李则徐/AppData/Local/uv/cache/archive-v0' -Directory | Where-Object { Test-Path (Join-Path "'$_.FullName '"'numpy-2.3.5.dist-info') -Or (Test-Path (Join-Path "'$_.FullName '"'scipy-1.17.0.dist-info')) } | Select-Object -ExpandProperty FullName"
```

**Aggregated Output:**

```text
2.3.5-cp312-cp312-win_amd64
.continuity
.pytest_cache
antigravity_2.26.0
antigravity_analysis
avesta_master_corpus
avesta_yasna
claude-code-2.1.302
contra_complete_chronicles
EM-FloatFormer
EM-FloatFormer-cli1-cttc
EM-FloatFormer-cli2-pluto
EM-FloatFormer-cli3-ofdm-stress
EM-FloatFormer-cli4-p12-stage5
EM-FloatFormer-p10
EM-FloatFormer-p9
EM-FloatFormer-stage3d
EM-FloatFormer-stage4
hellenic_psalter
images_extracted
julian_translation
media
neural_compiler
openclaw-repo
outreach_20_arsenal
outreach_50_remote
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
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
Test-Path : �Ҳ�����������ơ�Or��ƥ��Ĳ�����
����λ�� ��:2 �ַ�: 569
+ ... Test-Path (Join-Path $_.FullName 'numpy-2.3.5.dist-info') -Or (Test-P ...
+                                                               ~~~
    + CategoryInfo          : InvalidArgument: (:) [Test-Path]��ParameterBindingException
    + FullyQualifiedErrorId : NamedParameterNotFound,Microsoft.PowerShell.Commands.TestPathCommand
 
```

### Event #27: `item.completed` — `command_execution` (`item_24`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$base='"'C:/Users/李则徐/AppData/Local/uv/cache'; Get-ChildItem \""'$base/archive-v0" -Directory | Where-Object { (Test-Path (Join-Path $_.FullName '"'numpy-2.3.5.dist-info')) -or (Test-Path (Join-Path "'$_.FullName '"'scipy-1.17.0.dist-info')) } | Select-Object -ExpandProperty FullName; Get-ChildItem \""'$base/environments-v2/1093dab72d22fcbe" -Force -Name; Get-Item "$base/wheels-v6/pypi/numpy/2.3.5-cp312-cp312-win_amd64" | Format-List Name,Mode,Length; Get-ChildItem '"'research/stage4_math_and_design/discussions/M4' -Name"
```

**Aggregated Output:**

```text
C:\Users\������\AppData\Local\uv\cache\archive-v0\3Oxde2-Jkv6KGjnQ
C:\Users\������\AppData\Local\uv\cache\archive-v0\cSnmXl2iWBpNAXGE
27fef6b82521218f
44669b92cf8db157
76580b3e21535888
7829de497d478757
7d6c70a4e0ec676a
8ab8c010e9f0c738
a4a0d4059abfef7e
fff2ded241c488c1


Name   : 2.3.5-cp312-cp312-win_amd64
Mode   : -a----
Length : 27



prompt_r1.md
round1_events.jsonl
round1_stderr.log
```

### Event #28: `item.completed` — `command_execution` (`item_25`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "Get-Content -TotalCount 40 research/stage4_math_and_design/discussions/M4/prompt_r1.md; Get-ChildItem 'C:/Users/李则徐/AppData/Local/uv/cache/simple-v21' -Name; Test-Path 'C:/376381/miniconda3/python312.dll'; Test-Path 'C:/114514/python312.dll'; rg --files --hidden 'C:/Users/李则徐/AppData/Local/uv/cache/simple-v21' | rg '(numpy|scipy)'"
```

**Aggregated Output:**

````text
# Track M4 (Round 1): Real-Data Distribution-Shift Bounds, Pluto/CTTC Benchmark Adaptation & Modal Free-Tier Architecture

You are **Track M4**, part of a 4-track mathematical & architectural multi-agent collaboration preparing **EM-FloatFormer Stage 4** (`docs/LONG_TERM_RESEARCH_PLAN.md` Stage 4: real SDR data validation on ADALM-PLUTO 850 MHz + CTTC 433/708/2450 MHz paired with frozen Qwen3.5-4B / fixed encoder alignment).

## 0. Strict Workspace, Environment, and Immutability Rules
- **Repository Root**: `C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4`
- **Your Writable Directory**: `research/stage4_math_and_design/M4/` ONLY.
- **FORBIDDEN Edits**: Do NOT modify or delete any files outside `research/stage4_math_and_design/M4/`. (To inspect real-data code on remote git branches, use read-only `git show origin/research/...` �?do NOT check out other branches or alter git HEAD!)
- **Python Environment Guardrail**: System/Anaconda `python` crashes with MKL `libiomp5md.dll` conflicts on this machine. **ALWAYS** execute Python scripts via:
  ```powershell
  uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B <script.py>
  ```

## 1. Grounded Context to Inspect First (Read-Only)
1. `docs/LONG_TERM_RESEARCH_PLAN.md` (especially Stage 4 specification & Phase goals)
2. `research/matched_cache/learned.py` (`equivariant_b8_s17` input assumption: 3 complex samples `(3,)` at unit-energy QPSK + AWGN SNR=2 dB)
3. `research/stage3d_qwen_signed_phase/PROTOCOL_LOCK.json` and `REPORT.md` (budget note: `$17.12` spent on Stage 3D out of `$30.00` Modal free credit, **`~$12.88` remaining free compute credits**, `Billed Cost: $0.00` hard constraint!)
4. Read-only `git show` on existing real-data branches in this repo:
   - `git show origin/research/realdata-pluto-qpsk:research/realdata_pluto_qpsk/PROTOCOL.md`
   - `git show origin/research/realdata-pluto-qpsk:research/realdata_pluto_qpsk/prepare_benchmarks.py`
   - `git show origin/research/realdata-pluto-qpsk:research/realdata_pluto_qpsk/waveform_frontend.py`
   - `git show origin/research/realdata-pluto-qpsk:research/realdata_pluto_qpsk/receivers.py`
   - `git show origin/research/realdata-pluto-qpsk:research/realdata_pluto_qpsk/download_data.py`
   - `git show origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/PROTOCOL.md`
   - `git show origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/adapter.py`
   - `git show origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/experiments.py`
   - `git show origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/download_cttc.py`

### Summary of Existing Real-Data Physics in This Repo
- **ADALM-PLUTO (`QPSKlink.mat`, Mendeley `10.17632/k72b4pnc5g.1`, `822.76 MB`)**:
  - `7,096` frames × `7,560` raw IQ samples (`SPS = 20`, 126 QPSK symbols × 3 copies; Sync `[0:13]`, Online Pilot `[13:21]`, Holdout Probe `[21:26]`, Payload `[26:126]`).
  - Hardware AC-coupling / zero-IF DC cancellation $y_t[k] = h_t(s_t[k] - \bar{s}_t) + \text{ISI} + w_t[k]$ ($r = 0.999$), fading $|h_t| \in [0.0091, 0.0308]$ (`>10.6 dB`), 20-frame hardware batch jumps (p90 `36.4%`, max `0.94 rad`), `76.4%` remote recurrence, plus 8 real/stress regimes (`raw_original`, `awgn_snr{10,6,2,-2}db`, `cfo_0p015_raw`, `cfo_0p020_snr6db`, `phase_jump_snr6db`). Zero-leakage sequential splits: `train [200,3200)`, `dev [3500,4500)`, `test [4800,7040)` (`G = 16` clusters × 140 frames), independent warmup pools (`200` frames each) and `100`-frame guard bands.
- **CTTC USRP X310 (`10.5281/zenodo.4895133`, 9 captures `3 bands × 3 SNRs`, `~1.9 GB`)**:
  - `256`-symbol packets (`SPS = 4`, RRC match-filtered `file_2.dat`; `KNOWN_IDX` = 10 symbols, `HOLDOUT_IDX` = 16 symbols, `PAYLOAD_IDX` = 230 symbols).
  - Real crystal SFO (`+0.39..+0.55 ppm`), large carrier CFO (`-1285..-158 Hz`), `-3.22 dB` (`f708_snr0`) to `+24.01 dB` (`f708_snr20`), hardware coherent spur (`0.260 @ +1503 Hz` on `f708_snr10`), EVEN/ODD bit offset (`f433_snr20/snr0` ODD=1).
  - Zero-leakage cyclic source isolation (`>= 4,096`-symbol guards, `G = 72` session-loop clusters; `train=1440, dev=864, test=1440` + `1080` stress packets).

## 2. Mathematical & Architectural Questions You Must Answer Rigorously
1. **Distribution-Shift & Misspecification Bounds for Frozen `equivariant_b8_s17` on Real Hardware Channels**:
   `equivariant_b8_s17` (`learned.py`) takes 3 complex samples per symbol window `(y[0], y[1], y[2])` calibrated for unit-amplitude `QPSK/sqrt(3) + CN(0, sigma_2dB^2)` (`sigma_2dB^2 = 1 / 10^0.2 = 0.63096`). On Pluto & CTTC, matched-filtered symbol observations are scalar complex symbols $\tilde{y}_k = h_k s_k + d_k + w_k$ at arbitrary amplitude $|h_k| \in [0.009, 0.031]$ (or arbitrary USRP gain) and arbitrary SNR (`-3.2 dB` to `+24 dB`), OR windowed groups of symbols!
pypi
False
False
C:/Users/?????AppData/Local/uv/cache/simple-v21\pypi\numpy.rkyv
C:/Users/?????AppData/Local/uv/cache/simple-v21\pypi\numpy.lock
C:/Users/?????AppData/Local/uv/cache/simple-v21\pypi\scipy.rkyv
C:/Users/?????AppData/Local/uv/cache/simple-v21\pypi\scipy.lock
````

### Event #29: `item.completed` — `command_execution` (`item_26`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$src='"'C:/Users/李则徐/AppData/Local/uv/cache'; "'$dst='"'research/stage4_math_and_design/M4/.uv-m4'; New-Item -ItemType Directory -Force -Path \""'$dst/archive-v0","$dst/wheels-v6/pypi","$dst/simple-v21/pypi" | Out-Null; foreach ($p in '"'3Oxde2-Jkv6KGjnQ','cSnmXl2iWBpNAXGE') { Copy-Item -LiteralPath \""'$src/archive-v0/$p" -Destination "$dst/archive-v0" -Recurse }; foreach ($p in '"'numpy','scipy') { Copy-Item -LiteralPath \""'$src/wheels-v6/pypi/$p" -Destination "$dst/wheels-v6/pypi" -Recurse; Copy-Item -LiteralPath "$src/simple-v21/pypi/$p.rkyv" -Destination "$dst/simple-v21/pypi" }; $env:UV_CACHE_DIR=(Resolve-Path $dst).Path; $env:UV_OFFLINE='"'1'; "'$env:UV_NO_PROJECT='"'1'; uv python find 3.12"
```

**Aggregated Output:**

```text
C:\qazwsx\python.exe
```

### Event #30: `item.completed` — `command_execution` (`item_27`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n 'def bits_to_qpsk|CONST' research/realdata_cttc_qpsk -g '*.py'; git grep -n -e 'def bits_to_qpsk' -e 'stress_packets' -e 'stress.*360' origin/research/realdata-cttc-indoor-qpsk -- research/realdata_cttc_qpsk; Get-Content -Encoding utf8 research/stage3d_qwen_signed_phase/modal_runner.py | Select-Object -Skip 58 -First 45"
```

**Aggregated Output:**

```text
rg: research/realdata_cttc_qpsk: IO error for operation on research/realdata_cttc_qpsk: 系统找不到指定的文件。 (os error 2)
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/PROTOCOL.md:52:   - 独立可控物理扰动压力测试（在原始实测 IQ 上注入 3 类扰动：`stress_low_snr`、`stress_cfo_phase_jump`、`stress_burst_impairment`）：`train = 720`、`dev = 432`、`test = 1,080` 包（每类扰动 360 包，覆盖全部 9 会话与 72 簇），结果与原始实测完全分轨保存。
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/evaluator.py:40:def bits_to_qpsk(bits: np.ndarray) -> np.ndarray:
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:25:f2450_snr20_QPSK__L01__TT010__S394544__stress_burst_impairment,f2450_snr20_QPSK,2450,20,f2450_snr20_QPSK__L01,classical_mmse,True,2.11695,0.18261,0.09565,1,0.5149,-2.8828,-0.33185,0.29439,0.07842,230,42,44,0.33333,0.16667,-1.9363,0.25,0.125,-4.52012,0.39927,0.13609,-12.8239,0.94352,-881.0142,265.42658,0.15478
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:43:f2450_snr20_QPSK__L02__TT014__S418672__stress_burst_impairment,f2450_snr20_QPSK,2450,20,f2450_snr20_QPSK__L02,classical_mmse,True,2.23499,0.1913,0.10217,1,0.41648,-3.80406,-1.75059,0.28388,0.07869,230,44,47,0.33333,0.16667,-0.35403,0,0,-6.5221,0.36046,0.04891,-12.83663,0.72707,-1005.30606,-10.65368,1
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:67:f2450_snr20_QPSK__L04__TT006__S370416__stress_burst_impairment,f2450_snr20_QPSK,2450,20,f2450_snr20_QPSK__L04,classical_mmse,True,2.21086,0.68696,0.44783,1,1.3606,1.33732,26.53906,1.14889,0.34591,230,158,206,0.33333,0.16667,-1.08149,0,0,-2.73404,0.84548,0.83748,-12.13846,1.18557,1273.64986,2126.68164,0.1761
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:83:f2450_snr20_QPSK__L05__TT008__S382480__stress_low_snr,f2450_snr20_QPSK,2450,20,f2450_snr20_QPSK__L05,classical_mmse,True,2.24101,0.36087,0.19348,1,0.54397,-2.64428,-1.35372,0.39046,0.12526,230,83,89,0,0,-1.86648,0.25,0.125,-1.26596,0.56531,0.21817,-12.10492,1.00476,-1342.5674,-112.32753,-0.00425
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:124:f2450_snr10_QPSK__L00__TT002__S346288__stress_burst_impairment,f2450_snr10_QPSK,2450,10,f2450_snr10_QPSK__L00,classical_mmse,True,5.31659,0.29565,0.1587,1,0.51235,-2.90429,-0.42234,0.3601,0.10962,230,68,73,0.33333,0.16667,-1.32052,0,0,-8.32872,0.33235,0.06468,-15.48908,0.60476,-987.21139,-147.48578,1.05046
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:265:f2450_snr0_QPSK__L01__TT010__S394544__stress_burst_impairment,f2450_snr0_QPSK,2450,0,f2450_snr0_QPSK__L01,classical_mmse,True,1.73603,0.36087,0.19348,1,0.56944,-2.44551,0.90125,0.41205,0.1269,230,83,89,0.16667,0.08333,-4.00269,0.25,0.125,-2.13944,0.71176,0.07995,-16.49579,0.94414,-941.18843,-48.52173,-0.10962
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:276:f2450_snr0_QPSK__L02__TT005__S364384__stress_cfo_phase_jump,f2450_snr0_QPSK,2450,0,f2450_snr0_QPSK__L02,classical_mmse,True,1.93053,0.26957,0.13478,1,0.47005,-3.27858,-0.95473,0.33986,0.10181,230,62,62,0,0,-7.36034,0.25,0.125,-0.04761,0.35827,0.00803,-17.46832,0.60078,64.52589,709.36468,0.45241
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:286:f2450_snr0_QPSK__L02__TT018__S442800__stress_burst_impairment,f2450_snr0_QPSK,2450,0,f2450_snr0_QPSK__L02,classical_mmse,True,2.16709,0.36087,0.21087,1,0.64798,-1.88437,2.50712,0.47051,0.14771,230,83,97,0.5,0.25,0.30993,0,0,-5.555,0.53064,0.156,-16.8926,0.39052,-1323.26415,13.4466,-0.45104
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:317:f2450_snr0_QPSK__L05__TT000__S334224__stress_low_snr,f2450_snr0_QPSK,2450,0,f2450_snr0_QPSK__L05,classical_mmse,True,2.22935,0.36087,0.21957,1,0.64684,-1.89205,2.13136,0.46873,0.15332,230,83,101,0.33333,0.16667,-2.21343,0.25,0.125,-1.33035,0.52111,0.10427,-16.39877,0.65534,-1213.56455,60.08108,-0.03419
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:374:f708_snr20_QPSK__L00__TT016__S430736__stress_low_snr,f708_snr20_QPSK,708,20,f708_snr20_QPSK__L00,classical_mmse,True,2.26204,0.16087,0.08696,1,0.37853,-4.21905,-3.205,0.21726,0.06239,230,37,40,0,0,-4.81614,0.25,0.125,-3.27743,0.28702,0.04491,-12.72287,0.33608,-448.41775,-42.41128,0.54031
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:385:f708_snr20_QPSK__L01__TT010__S394544__stress_burst_impairment,f708_snr20_QPSK,708,20,f708_snr20_QPSK__L01,classical_mmse,True,2.27667,0.17391,0.09348,1,0.40987,-3.87358,-1.60828,0.25816,0.06672,230,40,43,0,0,-6.29791,0,0,-8.18651,0.36049,0.02183,-13.01292,0.64088,-516.24542,107.94392,0.84754
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:422:f708_snr20_QPSK__L04__TT000__S334224__stress_low_snr,f708_snr20_QPSK,708,20,f708_snr20_QPSK__L04,classical_mmse,True,2.31209,0.27391,0.14565,1,0.49503,-3.05365,-0.61266,0.34608,0.10828,230,63,67,0,0,-11.47004,0.25,0.125,-2.48834,0.33605,0.10712,-12.55079,0.60659,-449.65507,-33.6486,0.0336
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:455:f708_snr20_QPSK__L06__TT004__S358352__stress_low_snr,f708_snr20_QPSK,708,20,f708_snr20_QPSK__L06,classical_mmse,True,6.66055,0.21304,0.10652,1,0.43336,-3.6315,-1.64388,0.2783,0.08411,230,49,49,0.16667,0.08333,-2.03505,0,0,-8.83498,0.2844,0.03288,-12.41431,0.62727,-360.54468,55.46179,0.02713
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:457:f708_snr20_QPSK__L06__TT006__S370416__stress_burst_impairment,f708_snr20_QPSK,708,20,f708_snr20_QPSK__L06,classical_mmse,True,4.71497,0.33478,0.17826,1,0.6115,-2.13602,1.69488,0.52086,0.13512,230,77,82,0.16667,0.08333,-4.1105,0,0,-8.6264,0.55848,0.12176,-12.02155,1.33712,27.84608,235.58553,-0.82489
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:482:f708_snr10_QPSK__L00__TT000__S334224__stress_low_snr,f708_snr10_QPSK,708,10,f708_snr10_QPSK__L00,classical_mmse,True,4.80826,0.43913,0.24348,1,0.67145,-1.72984,2.13976,0.50068,0.16347,230,101,112,0.16667,0.08333,-2.53182,0.25,0.125,-1.34523,0.76425,0.21376,-14.30617,1.24367,111.52053,270.36086,-1.01583
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:491:f708_snr10_QPSK__L00__TT012__S406608__stress_low_snr,f708_snr10_QPSK,708,10,f708_snr10_QPSK__L00,classical_mmse,True,1.63728,0.36087,0.19348,1,0.59547,-2.25142,1.47828,0.43544,0.14008,230,83,89,0.5,0.25,0.91891,0,0,-5.75612,0.61886,0.04832,-14.93881,0.71498,-351.74971,-87.90937,-0.8153
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:521:f708_snr10_QPSK__L02__TT012__S406608__stress_low_snr,f708_snr10_QPSK,708,10,f708_snr10_QPSK__L02,classical_mmse,True,1.96275,0.31304,0.16739,1,0.51672,-2.86748,-0.48829,0.36038,0.1154,230,72,77,0,0,-10.25393,0.25,0.125,-0.72011,0.40178,0.16279,-15.21393,0.67239,-232.66128,16.17905,-0.25344
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:541:f708_snr10_QPSK__L03__TT018__S442800__stress_burst_impairment,f708_snr10_QPSK,708,10,f708_snr10_QPSK__L03,classical_mmse,True,1.95317,0.24348,0.13696,1,0.47942,-3.19286,-0.38119,0.33646,0.10277,230,56,63,0.16667,0.16667,-1.26088,0.25,0.125,-4.38646,0.36037,0.06956,-15.25163,0.73295,-545.61597,-151.82634,0.68916
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:545:f708_snr10_QPSK__L04__TT004__S358352__stress_low_snr,f708_snr10_QPSK,708,10,f708_snr10_QPSK__L04,classical_mmse,True,1.9548,0.32609,0.17174,1,0.54152,-2.66389,0.07697,0.39156,0.12542,230,75,79,0.16667,0.08333,-3.30963,0,0,-4.55376,0.60072,0.1835,-14.84893,1.00754,-31.07592,207.76441,-0.13603
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:549:f708_snr10_QPSK__L04__TT009__S388512__stress_cfo_phase_jump,f708_snr10_QPSK,708,10,f708_snr10_QPSK__L04,classical_mmse,True,1.93538,0.1087,0.05435,1,0.29103,-5.36063,-4.74339,0.16965,0.04874,230,25,25,0,0,-7.88493,0,0,-7.15237,0.22419,0.15804,-16.29764,1.37096,13.09482,474.70188,0.41837
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:552:f708_snr10_QPSK__L04__TT013__S412640__stress_cfo_phase_jump,f708_snr10_QPSK,708,10,f708_snr10_QPSK__L04,classical_mmse,True,1.92746,0.0087,0.00435,1,0.19405,-7.12083,-7.36029,0.01696,0.00464,230,2,2,0,0,-11.42838,0,0,-9.12503,0.18985,0.08654,-16.45204,1.34183,-760.77902,77.28887,0.27185
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:584:f708_snr10_QPSK__L06__TT016__S430736__stress_low_snr,f708_snr10_QPSK,708,10,f708_snr10_QPSK__L06,classical_mmse,True,5.06685,0.36087,0.18913,1,0.56942,-2.44565,1.20945,0.42314,0.13377,230,83,87,0.16667,0.08333,-1.79951,0,0,-6.6991,0.71395,0.16054,-14.48583,1.36453,-569.7365,-390.89617,1.125
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:645:f708_snr0_QPSK__L02__TT017__S436768__stress_cfo_phase_jump,f708_snr0_QPSK,708,0,f708_snr0_QPSK__L02,classical_mmse,True,5.09353,0.62174,0.36087,1,0.95498,-0.20004,9.91293,0.70936,0.24108,230,143,166,0.33333,0.16667,-3.09506,0.25,0.125,1.54769,0.84142,0.6866,-17.22726,0.93248,1367.80555,1358.79351,0.495
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:663:f708_snr0_QPSK__L04__TT001__S340256__stress_cfo_phase_jump,f708_snr0_QPSK,708,0,f708_snr0_QPSK__L04,classical_mmse,True,1.77701,0.57391,0.34348,1,0.84684,-0.72198,4.0158,0.61054,0.21129,230,132,158,0.5,0.25,1.85719,0.5,0.25,3.36003,0.7995,0.45584,-17.24781,0.88797,-760.95339,275.34262,1.125
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:733:f433_snr20_QPSK__L00__TT014__S418672__stress_burst_impairment,f433_snr20_QPSK,433,20,f433_snr20_QPSK__L00,classical_mmse,True,2.37115,0.21304,0.11087,1,0.43601,-3.60503,-1.34292,0.30823,0.08304,230,49,51,0,0,-6.73788,0,0,-8.56639,0.39992,0.01021,-12.83831,1.00906,211.04772,69.59963,0.03035
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:747:f433_snr20_QPSK__L01__TT013__S412640__stress_cfo_phase_jump,f433_snr20_QPSK,433,20,f433_snr20_QPSK__L01,classical_mmse,True,2.25733,0.16957,0.08478,1,0.36001,-4.43685,-3.02609,0.24008,0.07155,230,39,39,0.16667,0.08333,-4.25027,0,0,-7.91444,0.16972,0.04203,-13.42504,0.54755,-1275.7705,-588.63355,0.17877
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:794:f433_snr20_QPSK__L04__TT016__S430736__stress_low_snr,f433_snr20_QPSK,433,20,f433_snr20_QPSK__L04,classical_mmse,True,2.45161,0.36087,0.19565,1,0.54428,-2.64175,0.66368,0.39945,0.131,230,83,90,0,0,-6.09592,0.25,0.125,-1.87521,0.63602,0.00965,-12.40833,0.98415,-199.09362,-31.24694,-0.18347
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:801:f433_snr20_QPSK__L05__TT005__S364384__stress_cfo_phase_jump,f433_snr20_QPSK,433,20,f433_snr20_QPSK__L05,classical_mmse,True,2.33318,0.23478,0.11739,1,0.39336,-4.05206,-2.61357,0.27545,0.08481,230,54,54,0.16667,0.08333,-7.03258,0.25,0.125,-1.87817,0.23602,0.0832,-13.16754,0.69308,-1146.48301,-493.62434,0.58917
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:803:f433_snr20_QPSK__L05__TT008__S382480__stress_low_snr,f433_snr20_QPSK,433,20,f433_snr20_QPSK__L05,classical_mmse,True,2.26787,0.38696,0.21957,1,0.60064,-2.21387,1.71711,0.44107,0.14607,230,89,101,0.16667,0.16667,0.48721,0.25,0.125,-3.84427,0.58859,0.05718,-12.29631,0.85546,-330.89587,-168.0492,-0.93609
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:821:f433_snr20_QPSK__L06__TT012__S406608__stress_low_snr,f433_snr20_QPSK,433,20,f433_snr20_QPSK__L06,classical_mmse,True,2.29821,0.30435,0.16087,1,0.5429,-2.65276,0.2505,0.39522,0.12399,230,70,74,0.5,0.25,0.24037,0.25,0.125,-1.83601,0.31428,0.06131,-12.66786,0.33131,18.48001,191.32669,1.125
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:836:f433_snr20_QPSK__L07__TT012__S406608__stress_low_snr,f433_snr20_QPSK,433,20,f433_snr20_QPSK__L07,classical_mmse,True,2.29192,0.36087,0.19348,1,0.53464,-2.71938,-0.14954,0.39955,0.1286,230,83,89,0,0,-7.16982,0,0,-3.89877,0.43073,0.17944,-12.50153,1.0208,95.1726,273.01928,0.98778
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:837:f433_snr20_QPSK__L07__TT013__S412640__stress_cfo_phase_jump,f433_snr20_QPSK,433,20,f433_snr20_QPSK__L07,classical_mmse,True,2.33355,0.06957,0.03478,1,0.23882,-6.21938,-5.40575,0.09183,0.0267,230,16,16,0,0,-8.1607,0,0,-10.78576,0.28412,0.03953,-13.33079,1.03696,-76.16918,-103.36085,0.00791
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:845:f433_snr10_QPSK__L00__TT004__S358352__stress_low_snr,f433_snr10_QPSK,433,10,f433_snr10_QPSK__L00,classical_mmse,True,4.63626,0.28261,0.15652,1,0.51849,-2.85255,-0.091,0.36085,0.11266,230,65,72,0,0,-5.81445,0.25,0.125,-0.94354,0.46584,0.06147,-14.78248,0.72981,-334.35884,-100.01375,0.95425
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:895:f433_snr10_QPSK__L03__TT010__S394544__stress_burst_impairment,f433_snr10_QPSK,433,10,f433_snr10_QPSK__L03,classical_mmse,True,2.06964,0.26957,0.14565,1,0.5252,-2.79676,-0.1401,0.36075,0.10704,230,62,67,0,0,-7.17025,0.5,0.375,0.34594,0.37708,0.07588,-14.96485,0.85268,-352.79506,15.76459,-0.01344
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:958:f433_snr10_QPSK__L07__TT014__S418672__stress_burst_impairment,f433_snr10_QPSK,433,10,f433_snr10_QPSK__L07,classical_mmse,True,2.24039,0.23478,0.12609,1,0.48789,-3.1168,-0.99199,0.32692,0.09683,230,54,58,0.16667,0.08333,-3.1597,0,0,-7.53602,0.29917,0.16673,-14.94927,0.70561,-390.75787,84.36047,0.0067
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:994:f433_snr0_QPSK__L02__TT002__S346288__stress_burst_impairment,f433_snr0_QPSK,433,0,f433_snr0_QPSK__L02,classical_mmse,True,2.29031,0.36087,0.19783,1,0.59192,-2.27735,1.58003,0.42491,0.13395,230,83,91,0.16667,0.08333,-4.41185,0.25,0.125,-0.18855,0.58943,0.00949,-16.25261,0.85578,-347.37802,79.09356,0.30027
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:997:f433_snr0_QPSK__L02__TT006__S370416__stress_burst_impairment,f433_snr0_QPSK,433,0,f433_snr0_QPSK__L02,classical_mmse,True,2.25404,0.36087,0.1913,1,0.55545,-2.55358,0.64304,0.40476,0.13026,230,83,88,0.5,0.25,-0.53249,0,0,-6.17979,0.65921,0.12949,-15.80653,1.39629,-278.93819,245.39343,1
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:1004:f433_snr0_QPSK__L02__TT016__S430736__stress_low_snr,f433_snr0_QPSK,433,0,f433_snr0_QPSK__L02,classical_mmse,True,2.27092,0.36087,0.20217,1,0.62332,-2.05286,1.98739,0.45631,0.14898,230,83,93,0.33333,0.16667,0.45734,0,0,-7.73862,0.58394,0.10243,-16.28485,0.50096,100.86879,289.17873,0.16778
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:1018:f433_snr0_QPSK__L03__TT014__S418672__stress_burst_impairment,f433_snr0_QPSK,433,0,f433_snr0_QPSK__L03,classical_mmse,True,2.22821,0.34783,0.2,1,0.66137,-1.79554,1.96327,0.47408,0.14854,230,80,92,0.33333,0.16667,-3.5056,0.25,0.125,-2.46733,0.64193,0.13603,-16.22191,0.98208,-91.43593,-61.38582,1.125
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:1050:f433_snr0_QPSK__L05__TT017__S436768__stress_cfo_phase_jump,f433_snr0_QPSK,433,0,f433_snr0_QPSK__L05,classical_mmse,True,2.36079,0.23043,0.11522,1,0.40934,-3.87914,-1.92219,0.2942,0.0894,230,53,53,0,0,-6.10355,0.25,0.125,-3.38385,0.31664,0.06308,-17.43007,0.66559,-460.33294,-566.93614,-0.0192
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:1051:f433_snr0_QPSK__L05__TT018__S442800__stress_burst_impairment,f433_snr0_QPSK,433,0,f433_snr0_QPSK__L05,classical_mmse,True,2.3604,0.3913,0.21087,1,0.68159,-1.6648,2.72821,0.50513,0.16075,230,90,97,0.33333,0.16667,-0.47446,0.25,0.125,-3.06516,0.57781,0.19682,-15.92135,0.70915,-684.76753,-110.29472,1.125
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:1060:f433_snr0_QPSK__L06__TT010__S394544__stress_burst_impairment,f433_snr0_QPSK,433,0,f433_snr0_QPSK__L06,classical_mmse,True,2.29646,0.36522,0.19783,1,0.62234,-2.05975,2.08349,0.45557,0.14073,230,84,91,0.5,0.25,-0.77185,0,0,-4.56442,0.58644,0.0704,-16.13806,0.82686,-360.10714,-264.27311,0.79018
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:1081:f433_snr0_QPSK__L07__TT018__S442800__stress_burst_impairment,f433_snr0_QPSK,433,0,f433_snr0_QPSK__L07,classical_mmse,True,1.67117,0.36087,0.19348,1,0.60725,-2.16631,1.85368,0.44046,0.13701,230,83,89,0.33333,0.16667,-3.43927,0.25,0.125,-2.95642,0.47968,0.11719,-16.42652,0.54792,-168.84346,-92.44208,1.02923
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:1105:f2450_snr20_QPSK__L01__TT010__S394544__stress_burst_impairment,f2450_snr20_QPSK,2450,20,f2450_snr20_QPSK__L01,A4_global_content_attn,True,3.55844,0.12609,0.06739,1,0.43687,-3.5965,-1.63346,0.23604,0.05988,230,29,31,0.33333,0.16667,-1.91778,0.25,0.125,-4.32592,0.38668,0.1112,-12.83348,1.00457,-1169.70512,-23.26434,0.25913
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:1130:f2450_snr20_QPSK__L03__TT004__S358352__stress_low_snr,f2450_snr20_QPSK,2450,20,f2450_snr20_QPSK__L03,A4_global_content_attn,True,2.58066,0.61304,0.36087,1,1.00522,0.0226,4.47292,0.73594,0.24381,230,141,166,0.5,0.25,-4.53984,0.75,0.375,-3.1076,0.85809,0.88219,-11.44714,2.06644,-259.40075,995.83911,-0.29743
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:1148:f2450_snr20_QPSK__L04__TT008__S382480__stress_low_snr,f2450_snr20_QPSK,2450,20,f2450_snr20_QPSK__L04,A4_global_content_attn,True,2.56484,0.26522,0.13261,1,0.47025,-3.27671,-1.05132,0.32599,0.10239,230,61,61,0.16667,0.08333,-6.27208,0,0,-4.52415,0.36032,0.05909,-12.72829,0.79387,-1208.15582,57.08405,0.97971
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:1169:f2450_snr20_QPSK__L05__TT016__S430736__stress_low_snr,f2450_snr20_QPSK,2450,20,f2450_snr20_QPSK__L05,A4_global_content_attn,True,3.41198,0.26957,0.14783,1,0.42687,-3.697,-1.34001,0.27232,0.0903,230,62,68,0,0,-7.64895,0.25,0.125,-11.24932,0.36039,0.05331,-12.70977,0.76993,-1284.6586,-74.41873,-0.05588
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:1223:f2450_snr10_QPSK__L01__TT008__S382480__stress_low_snr,f2450_snr10_QPSK,2450,10,f2450_snr10_QPSK__L01,A4_global_content_attn,True,3.34694,0.33478,0.17826,1,0.53898,-2.68425,-2.02468,0.39279,0.12444,230,77,82,0.33333,0.25,0.14988,0,0,-2.73212,0.36085,0.20982,-15.30625,0.84997,-1202.96538,-4.94597,0.94216
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:1307:f2450_snr10_QPSK__L07__TT000__S334224__stress_low_snr,f2450_snr10_QPSK,2450,10,f2450_snr10_QPSK__L07,A4_global_content_attn,True,4.80935,0.36087,0.2,1,0.57756,-2.38401,-1.097,0.43337,0.13927,230,83,92,0.16667,0.08333,-1.43775,0.25,0.125,-2.52143,0.48572,0.18998,-15.16974,0.8986,-1253.32929,-45.30988,1
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:1324:f2450_snr0_QPSK__L00__TT002__S346288__stress_burst_impairment,f2450_snr0_QPSK,2450,0,f2450_snr0_QPSK__L00,A4_global_content_attn,True,3.30659,0.36087,0.21739,1,0.63405,-1.97878,2.34781,0.46095,0.14805,230,83,100,0.66667,0.33333,3.80337,0.5,0.25,1.8019,0.53585,0.0515,-16.70456,0.78587,-1245.8689,-131.20607,0.0113
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:1345:f2450_snr0_QPSK__L01__TT010__S394544__stress_burst_impairment,f2450_snr0_QPSK,2450,0,f2450_snr0_QPSK__L01,A4_global_content_attn,True,3.45288,0.36087,0.19348,1,0.57012,-2.4403,0.91626,0.41475,0.12716,230,83,89,0.16667,0.08333,-4.00521,0.25,0.125,-2.08719,0.67912,0.04916,-16.54606,0.99206,-931.18843,-38.52173,-0.10962
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:1366:f2450_snr0_QPSK__L02__TT018__S442800__stress_burst_impairment,f2450_snr0_QPSK,2450,0,f2450_snr0_QPSK__L02,A4_global_content_attn,True,3.82826,0.36087,0.2087,1,0.64211,-1.92394,2.53297,0.48793,0.14903,230,83,96,0.5,0.25,0.30819,0,0,-5.63674,0.53512,0.17534,-17.04399,0.52246,-1297.18616,39.52459,-0.45104
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:1381:f2450_snr0_QPSK__L03__TT018__S442800__stress_burst_impairment,f2450_snr0_QPSK,2450,0,f2450_snr0_QPSK__L03,A4_global_content_attn,True,2.98465,0.36087,0.20217,1,0.63114,-1.99876,2.25981,0.49397,0.14574,230,83,93,0.16667,0.08333,-3.66611,0.25,0.25,-2.21172,0.61578,0.20599,-16.3667,1.17848,-1168.65058,226.80483,1.08503
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:1397:f2450_snr0_QPSK__L05__TT000__S334224__stress_low_snr,f2450_snr0_QPSK,2450,0,f2450_snr0_QPSK__L05,A4_global_content_attn,True,2.97246,0.36087,0.21957,1,0.63193,-1.99333,2.13313,0.47054,0.15336,230,83,101,0.5,0.25,-2.21379,0.25,0.125,-1.30578,0.52677,0.15125,-16.46149,0.90737,-1207.54713,66.0985,-0.03419
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:1419:f2450_snr0_QPSK__L06__TT009__S388512__stress_cfo_phase_jump,f2450_snr0_QPSK,2450,0,f2450_snr0_QPSK__L06,A4_global_content_attn,True,3.04169,0.16957,0.08913,1,0.37286,-4.28448,-2.93276,0.26831,0.07474,230,39,41,0.16667,0.08333,-4.19117,0.5,0.25,3.61796,0.43609,0.05567,-17.41941,0.68201,-1744.96003,-15.37087,0.68513
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:1478:f708_snr20_QPSK__L02__TT008__S382480__stress_low_snr,f708_snr20_QPSK,708,20,f708_snr20_QPSK__L02,A4_global_content_attn,True,3.13607,0.28261,0.14565,1,0.47475,-3.23536,-1.41186,0.35194,0.10754,230,65,67,0,0,-6.83438,0,0,-3.12045,0.51004,0.24534,-12.54476,0.79567,-290.84961,115.15686,0.91058
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:1531:f708_snr20_QPSK__L05__TT018__S442800__stress_burst_impairment,f708_snr20_QPSK,708,20,f708_snr20_QPSK__L05,A4_global_content_attn,True,2.73195,0.12609,0.06304,1,0.36334,-4.3969,-2.5143,0.20701,0.04955,230,29,29,0,0,-7.48698,0,0,-3.44772,0.36043,0.03315,-12.84355,0.78741,-408.7334,118.58105,0.35216
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:1532:f708_snr20_QPSK__L06__TT000__S334224__stress_low_snr,f708_snr20_QPSK,708,20,f708_snr20_QPSK__L06,A4_global_content_attn,True,2.74024,0.17391,0.09348,1,0.37512,-4.25834,-2.55774,0.24217,0.07205,230,40,43,0.33333,0.16667,-1.37552,0,0,-7.82508,0.35158,0.01112,-12.59741,0.72033,-407.36032,8.64615,0.34242
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:1569:f708_snr10_QPSK__L00__TT009__S388512__stress_cfo_phase_jump,f708_snr10_QPSK,708,10,f708_snr10_QPSK__L00,A4_global_content_attn,True,3.29906,0.71739,0.36087,1,0.92997,-0.31533,3.17135,0.72694,0.2434,230,165,166,0.5,0.25,-6.97912,0.75,0.375,-2.28809,0.82175,0.79314,-14.47385,2.23695,-150.58479,-282.61855,1.12499
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:1619:f708_snr10_QPSK__L03__TT016__S430736__stress_low_snr,f708_snr10_QPSK,708,10,f708_snr10_QPSK__L03,A4_global_content_attn,True,3.17517,0.4087,0.21304,1,0.59109,-2.28347,-0.40899,0.44186,0.1444,230,94,98,0.16667,0.08333,-2.18176,0.5,0.25,-1.09961,0.50154,0.26974,-14.87986,1.08188,-211.36047,137.47986,0.08813
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:1623:f708_snr10_QPSK__L04__TT001__S340256__stress_cfo_phase_jump,f708_snr10_QPSK,708,10,f708_snr10_QPSK__L04,A4_global_content_attn,True,5.29445,0.77826,0.38913,1,0.90969,-0.41108,6.11805,0.9305,0.28499,230,179,179,0,0,-8.36042,0.5,0.25,-0.10038,0.868,0.68446,-14.87847,1.85782,-1350.59707,-651.8279,1.08135
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:1625:f708_snr10_QPSK__L04__TT004__S358352__stress_low_snr,f708_snr10_QPSK,708,10,f708_snr10_QPSK__L04,A4_global_content_attn,True,3.34787,0.31304,0.16739,1,0.53024,-2.75529,0.00165,0.38835,0.12283,230,72,77,0.16667,0.08333,-3.31351,0,0,-4.69757,0.60105,0.17267,-14.84035,1.13271,-91.07592,147.76441,-0.13603
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:1653:f708_snr10_QPSK__L06__TT001__S340256__stress_cfo_phase_jump,f708_snr10_QPSK,708,10,f708_snr10_QPSK__L06,A4_global_content_attn,True,3.11105,0.2087,0.10435,1,0.36064,-4.42922,-3.28482,0.27279,0.07989,230,48,48,0,0,-10.6962,0.5,0.25,-0.84237,0.25741,0.12213,-15.90907,1.46873,-431.81369,-512.28864,0.48002
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:1676:f708_snr10_QPSK__L07__TT012__S406608__stress_low_snr,f708_snr10_QPSK,708,10,f708_snr10_QPSK__L07,A4_global_content_attn,True,2.93383,0.34348,0.18478,1,0.54767,-2.61485,-1.43895,0.41937,0.13168,230,79,85,0.16667,0.08333,-5.98617,0,0,-7.33657,0.51586,0.51388,-14.93303,1.26887,-360.0457,183.79463,0.59146
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:1683:f708_snr0_QPSK__L00__TT001__S340256__stress_cfo_phase_jump,f708_snr0_QPSK,708,0,f708_snr0_QPSK__L00,A4_global_content_attn,True,3.04463,0.76522,0.47826,1,1.11612,0.47712,29.53188,0.76853,0.28028,230,176,220,0.16667,0.08333,-1.32282,0.5,0.25,5.2352,0.95384,1.12197,-17.22887,0.77974,1360.08637,1121.76766,0.3757
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:1737:f708_snr0_QPSK__L03__TT013__S412640__stress_cfo_phase_jump,f708_snr0_QPSK,708,0,f708_snr0_QPSK__L03,A4_global_content_attn,True,2.72676,0.38696,0.22609,1,0.65812,-1.81698,2.5076,0.48294,0.15826,230,89,104,0.33333,0.16667,-3.13084,0.5,0.25,1.96379,0.53604,0.03552,-17.63578,0.60903,-235.45369,-112.4231,1.01941
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:1798:f708_snr0_QPSK__L07__TT014__S418672__stress_burst_impairment,f708_snr0_QPSK,708,0,f708_snr0_QPSK__L07,A4_global_content_attn,True,3.27648,0.48261,0.27391,1,0.74633,-1.27071,3.79657,0.54786,0.18468,230,111,126,0.33333,0.25,4.29619,0.25,0.125,-4.36022,1.14253,0.172,-17.4149,0.92448,-632.53178,-59.57127,0.11155
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:1874:f433_snr20_QPSK__L04__TT016__S430736__stress_low_snr,f433_snr20_QPSK,433,20,f433_snr20_QPSK__L04,A4_global_content_attn,True,3.03742,0.36087,0.19565,1,0.54465,-2.6388,0.65681,0.41089,0.1335,230,83,90,0,0,-6.09657,0.25,0.125,-1.85422,0.63639,0.00879,-12.39967,1.13369,-195.07457,-27.22789,-0.18347
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:1883:f433_snr20_QPSK__L05__TT008__S382480__stress_low_snr,f433_snr20_QPSK,433,20,f433_snr20_QPSK__L05,A4_global_content_attn,True,3.41534,0.38261,0.21739,1,0.60355,-2.19289,1.56302,0.45295,0.14866,230,88,100,0.16667,0.16667,0.49753,0.25,0.125,-4.28157,0.53957,0.04334,-12.38433,1.15218,-211.12032,-48.27364,-0.93609
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:1887:f433_snr20_QPSK__L05__TT013__S412640__stress_cfo_phase_jump,f433_snr20_QPSK,433,20,f433_snr20_QPSK__L05,A4_global_content_attn,True,5.69667,0.13478,0.06957,1,0.31677,-4.99258,-3.65927,0.17654,0.05068,230,31,32,0.5,0.25,0.78524,0.25,0.125,-7.61244,0.28955,0.15492,-13.3859,0.88651,-110.95352,-360.61985,1.12311
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:1907:f433_snr20_QPSK__L07__TT000__S334224__stress_low_snr,f433_snr20_QPSK,433,20,f433_snr20_QPSK__L07,A4_global_content_attn,True,3.09501,0.1913,0.10435,1,0.41375,-3.83262,-2.31303,0.27878,0.08311,230,44,48,0.16667,0.08333,-3.36925,0.25,0.125,-5.06057,0.48439,0.16998,-12.6359,0.90063,-360.62477,-182.77809,0.57761
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:1917:f433_snr20_QPSK__L07__TT013__S412640__stress_cfo_phase_jump,f433_snr20_QPSK,433,20,f433_snr20_QPSK__L07,A4_global_content_attn,True,3.35396,0.11304,0.05652,1,0.26048,-5.84233,-5.03227,0.12042,0.0359,230,26,26,0,0,-8.16038,0,0,-10.15072,0.30008,0.07131,-13.26,1.1929,-146.16918,-173.36085,0.00791
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:1994:f433_snr10_QPSK__L04__TT016__S430736__stress_low_snr,f433_snr10_QPSK,433,10,f433_snr10_QPSK__L04,A4_global_content_attn,True,3.26519,0.21304,0.11087,1,0.42003,-3.76715,-1.62664,0.27428,0.0827,230,49,51,0.33333,0.16667,-2.64334,0.25,0.125,-2.43607,0.40208,0.11124,-14.7009,1.12878,-360.62477,-176.27967,-0.00901
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:1996:f433_snr10_QPSK__L04__TT018__S442800__stress_burst_impairment,f433_snr10_QPSK,433,10,f433_snr10_QPSK__L04,A4_global_content_attn,True,3.20382,0.24783,0.13261,1,0.47722,-3.21279,-0.45032,0.34393,0.09913,230,57,61,0.16667,0.08333,-4.14124,0.25,0.125,-4.96064,0.28368,0.0966,-15.43601,0.58297,-134.66485,-311.98004,-0.28379
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:2006:f433_snr10_QPSK__L05__TT012__S406608__stress_low_snr,f433_snr10_QPSK,433,10,f433_snr10_QPSK__L05,A4_global_content_attn,True,3.26222,0.26522,0.13478,1,0.47665,-3.218,-0.80518,0.33165,0.10428,230,61,62,0.16667,0.08333,-5.62585,0.25,0.25,-0.40307,0.3609,0.076,-15.12982,0.68673,-403.90614,-209.56105,0.59884
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:2008:f433_snr10_QPSK__L05__TT014__S418672__stress_burst_impairment,f433_snr10_QPSK,433,10,f433_snr10_QPSK__L05,A4_global_content_attn,True,2.92355,0.3087,0.16957,1,0.57038,-2.43839,1.1984,0.43885,0.12532,230,71,78,0.16667,0.08333,-3.35266,0.25,0.125,-2.04239,0.53607,0.06077,-14.79825,1.08025,-178.3764,-142.8812,-0.89678
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:2026:f433_snr10_QPSK__L06__TT018__S442800__stress_burst_impairment,f433_snr10_QPSK,433,10,f433_snr10_QPSK__L06,A4_global_content_attn,True,3.1421,0.36087,0.19783,1,0.66277,-1.78639,1.37974,0.50383,0.14692,230,83,91,0.16667,0.08333,-6.14434,0.75,0.375,0.82027,0.47172,0.22774,-14.9148,1.1193,-167.14593,-368.08783,0.9703
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:2041:f433_snr10_QPSK__L07__TT018__S442800__stress_burst_impairment,f433_snr10_QPSK,433,10,f433_snr10_QPSK__L07,A4_global_content_attn,True,3.23101,0.22174,0.11739,1,0.49657,-3.04022,-0.20937,0.37607,0.09678,230,51,54,0.16667,0.08333,-2.93609,0.25,0.125,-4.63506,0.33829,0.02187,-15.10436,0.9894,-453.01455,25.94679,-0.0308
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:2062:f433_snr0_QPSK__L01__TT006__S370416__stress_burst_impairment,f433_snr0_QPSK,433,0,f433_snr0_QPSK__L01,A4_global_content_attn,True,3.26963,0.40435,0.22609,1,0.66568,-1.76736,2.89507,0.52546,0.16535,230,93,104,0.16667,0.16667,-2.52765,0.25,0.125,-3.2042,0.84476,0.24956,-15.43654,1.63601,-224.25135,307.36796,-0.76914
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:2085:f433_snr0_QPSK__L02__TT017__S436768__stress_cfo_phase_jump,f433_snr0_QPSK,433,0,f433_snr0_QPSK__L02,A4_global_content_attn,True,3.24463,0.29565,0.14783,1,0.48549,-3.1382,-0.7065,0.36001,0.10789,230,68,68,0,0,-6.20348,0,0,-4.9046,0.38384,0.23454,-17.06283,0.95249,-493.19523,-313.10635,0.226
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:2094:f433_snr0_QPSK__L03__TT009__S388512__stress_cfo_phase_jump,f433_snr0_QPSK,433,0,f433_snr0_QPSK__L03,A4_global_content_attn,True,3.25661,0.36087,0.18261,1,0.55066,-2.59116,-0.65053,0.46082,0.13504,230,83,84,0,0,-10.1117,0,0,-4.62766,0.76094,0.32889,-16.16164,1.97099,-306.2582,-615.79273,-0.02421
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:2098:f433_snr0_QPSK__L03__TT014__S418672__stress_burst_impairment,f433_snr0_QPSK,433,0,f433_snr0_QPSK__L03,A4_global_content_attn,True,2.95428,0.36087,0.20217,1,0.68867,-1.61986,2.18704,0.49679,0.15598,230,83,93,0.16667,0.08333,-3.49271,0.25,0.125,-3.62777,0.62859,0.18778,-16.24204,0.95415,-211.43593,-181.38582,1.125
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:2111:f433_snr0_QPSK__L04__TT012__S406608__stress_low_snr,f433_snr0_QPSK,433,0,f433_snr0_QPSK__L04,A4_global_content_attn,True,3.40123,0.36087,0.19783,1,0.61016,-2.14554,1.83763,0.44747,0.14398,230,83,91,0.16667,0.08333,-0.42327,0,0,-9.20602,0.49424,0.06733,-15.8707,1.00434,-248.42052,-95.11058,0.91493
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:2161:f433_snr0_QPSK__L07__TT018__S442800__stress_burst_impairment,f433_snr0_QPSK,433,0,f433_snr0_QPSK__L07,A4_global_content_attn,True,3.26214,0.36087,0.19348,1,0.60786,-2.16194,1.87384,0.45739,0.1389,230,83,89,0.33333,0.16667,-3.44128,0.25,0.125,-3.02601,0.50485,0.16494,-16.45782,0.677,-178.84346,-102.44208,1.02923
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:2311:f2450_snr10_QPSK__L01__TT018__S442800__stress_burst_impairment,f2450_snr10_QPSK,2450,10,f2450_snr10_QPSK__L01,B1_hard_select,True,30.54553,0.3,0.16304,1,0.53603,-2.70809,0.61967,0.43038,0.12085,230,69,75,0.33333,0.16667,-2.94707,0,0,-7.47911,0.79342,0.12868,-14.76361,1.43762,-1377.30418,-94.82363,-0.9455
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:2315:f2450_snr10_QPSK__L02__TT004__S358352__stress_low_snr,f2450_snr10_QPSK,2450,10,f2450_snr10_QPSK__L02,B1_hard_select,True,31.66819,0.28696,0.16304,1,0.50614,-2.95733,-0.12311,0.36033,0.11495,230,66,75,0.16667,0.08333,-3.54704,0,0,-3.72746,0.32534,0.04055,-15.39272,0.61124,-1249.24701,-46.2276,0.10244
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:2346:f2450_snr10_QPSK__L04__TT005__S364384__stress_cfo_phase_jump,f2450_snr10_QPSK,2450,10,f2450_snr10_QPSK__L04,B1_hard_select,True,28.89566,0.01739,0.0087,1,0.16644,-7.78743,-7.09245,0.02251,0.0068,230,4,4,0,0,-13.00674,0,0,-8.36012,0.14715,0.05254,-16.64376,1.03398,-1541.36188,-288.43885,0.66504
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:2354:f2450_snr10_QPSK__L04__TT016__S430736__stress_low_snr,f2450_snr10_QPSK,2450,10,f2450_snr10_QPSK__L04,B1_hard_select,True,34.03965,0.32609,0.17391,1,0.50632,-2.95574,-0.45166,0.36052,0.11831,230,75,80,0,0,-8.49894,0,0,-3.1511,0.86901,0.3022,-14.75815,1.26387,-1405.73516,-242.71575,1.125
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:2423:f2450_snr0_QPSK__L01__TT008__S382480__stress_low_snr,f2450_snr0_QPSK,2450,0,f2450_snr0_QPSK__L01,B1_hard_select,True,32.99494,0.35217,0.19783,1,0.59205,-2.27641,1.36012,0.43217,0.13817,230,81,91,0.33333,0.16667,2.51159,0,0,-6.39155,0.51314,0.02158,-16.69379,0.63573,-1187.72657,95.91906,-0.25183
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:2437:f2450_snr0_QPSK__L02__TT006__S370416__stress_burst_impairment,f2450_snr0_QPSK,2450,0,f2450_snr0_QPSK__L02,B1_hard_select,True,31.68013,0.36087,0.2,1,0.64784,-1.88531,1.73448,0.46498,0.14677,230,83,92,0.33333,0.16667,-1.02618,0,0,-4.26703,0.6409,0.16428,-16.21093,1.0574,-1224.06793,-138.56193,0.99726
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:2446:f2450_snr0_QPSK__L02__TT018__S442800__stress_burst_impairment,f2450_snr0_QPSK,2450,0,f2450_snr0_QPSK__L02,B1_hard_select,True,31.3418,0.36087,0.2087,1,0.64605,-1.89732,2.58986,0.48049,0.14915,230,83,96,0.5,0.25,0.3025,0,0,-5.54692,0.51694,0.17044,-17.005,0.45317,-1257.18616,79.52459,-0.45104
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:2458:f2450_snr0_QPSK__L03__TT014__S418672__stress_burst_impairment,f2450_snr0_QPSK,2450,0,f2450_snr0_QPSK__L03,B1_hard_select,True,34.1524,0.36087,0.20217,1,0.61893,-2.08356,2.09552,0.45793,0.14141,230,83,93,0.33333,0.16667,-2.7601,0.25,0.125,-5.52037,0.56142,0.04975,-16.16596,1.41476,-1216.77193,-67.78549,-0.0562
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:2477:f2450_snr0_QPSK__L05__TT000__S334224__stress_low_snr,f2450_snr0_QPSK,2450,0,f2450_snr0_QPSK__L05,B1_hard_select,True,31.60916,0.36087,0.21957,1,0.63707,-1.95813,2.13289,0.4655,0.1525,230,83,101,0.5,0.25,-2.21334,0.25,0.125,-1.30318,0.52599,0.14551,-16.4374,0.77451,-1207.54713,66.0985,-0.03419
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:2498:f2450_snr0_QPSK__L06__TT008__S382480__stress_low_snr,f2450_snr0_QPSK,2450,0,f2450_snr0_QPSK__L06,B1_hard_select,True,29.68753,0.36087,0.2087,1,0.56823,-2.45478,1.1483,0.41633,0.1351,230,83,96,0.5,0.25,2.04049,0,0,-6.83442,0.57528,0.01823,-16.60935,0.70515,-1514.33693,-125.6913,0.05141
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:2503:f2450_snr0_QPSK__L06__TT014__S418672__stress_burst_impairment,f2450_snr0_QPSK,2450,0,f2450_snr0_QPSK__L06,B1_hard_select,True,32.36086,0.34348,0.18913,1,0.61221,-2.13099,1.79493,0.44583,0.13278,230,79,87,0.33333,0.16667,-3.33711,0,0,-7.16056,0.51908,0.06467,-16.87624,0.80478,-1348.40796,196.01048,1.1067
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:2518:f2450_snr0_QPSK__L07__TT014__S418672__stress_burst_impairment,f2450_snr0_QPSK,2450,0,f2450_snr0_QPSK__L07,B1_hard_select,True,32.2975,0.37826,0.2,1,0.59043,-2.28831,1.5285,0.44438,0.14277,230,87,92,0.33333,0.25,-1.71994,0,0,-7.85399,0.86347,0.28575,-16.11074,1.51497,-1243.91362,-360.29617,0.47241
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:2555:f708_snr20_QPSK__L02__TT004__S358352__stress_low_snr,f708_snr20_QPSK,708,20,f708_snr20_QPSK__L02,B1_hard_select,True,39.32192,0.18696,0.09565,1,0.36039,-4.43233,-2.72015,0.22938,0.06697,230,43,44,0,0,-7.24781,0.5,0.25,-1.00698,0.44269,0.00546,-12.47832,0.8086,-396.05422,-25.04776,0.3894
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:2588:f708_snr20_QPSK__L04__TT008__S382480__stress_low_snr,f708_snr20_QPSK,708,20,f708_snr20_QPSK__L04,B1_hard_select,True,39.50909,0.17391,0.0913,1,0.36044,-4.43165,-2.65687,0.21619,0.06501,230,40,42,0,0,-5.573,0,0,-1.82951,0.38767,0.02094,-12.56239,0.71555,-475.85823,-89.85176,-0.09845
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:2612:f708_snr20_QPSK__L06__TT000__S334224__stress_low_snr,f708_snr20_QPSK,708,20,f708_snr20_QPSK__L06,B1_hard_select,True,34.6313,0.20435,0.1087,1,0.42666,-3.69917,-1.84388,0.27899,0.08408,230,47,50,0,0,-1.3762,0,0,-8.02676,0.39544,0.1683,-12.48826,0.73262,-377.36032,38.64615,0.34242
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:2642:f708_snr10_QPSK__L00__TT000__S334224__stress_low_snr,f708_snr10_QPSK,708,10,f708_snr10_QPSK__L00,B1_hard_select,True,31.05591,0.43913,0.24348,1,0.67196,-1.72659,2.14086,0.5183,0.16588,230,101,112,0.16667,0.08333,-2.53154,0.25,0.125,-1.34083,0.78665,0.23829,-14.21498,1.47962,111.52053,270.36086,-1.01583
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:2670:f708_snr10_QPSK__L01__TT017__S436768__stress_cfo_phase_jump,f708_snr10_QPSK,708,10,f708_snr10_QPSK__L01,B1_hard_select,True,26.3271,0.01304,0.00652,1,0.16185,-7.90891,-7.72069,0.02818,0.00654,230,3,3,0,0,-15.49868,0,0,-8.99497,0.13609,0.05866,-16.88646,0.93548,-111.82835,62.77544,0.15657
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:2678:f708_snr10_QPSK__L02__TT008__S382480__stress_low_snr,f708_snr10_QPSK,708,10,f708_snr10_QPSK__L02,B1_hard_select,True,36.7535,0.31304,0.16957,1,0.53843,-2.68869,-0.37423,0.39799,0.12547,230,72,78,0.16667,0.08333,-2.40958,0.25,0.125,-7.36079,0.47121,0.29819,-14.88526,1.02255,-391.98654,-38.1462,-0.49969
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:2679:f708_snr10_QPSK__L02__TT009__S388512__stress_cfo_phase_jump,f708_snr10_QPSK,708,10,f708_snr10_QPSK__L02,B1_hard_select,True,33.3453,0.01304,0.00652,1,0.15864,-7.99577,-7.36072,0.0163,0.00505,230,3,3,0,0,-11.15637,0,0,-8.12864,0.15908,0.03329,-16.86884,1.13059,-398.16591,164.30734,0.19659
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:2699:f708_snr10_QPSK__L03__TT016__S430736__stress_low_snr,f708_snr10_QPSK,708,10,f708_snr10_QPSK__L03,B1_hard_select,True,34.9534,0.42174,0.21957,1,0.59392,-2.26275,-0.38705,0.4415,0.14494,230,97,101,0.16667,0.08333,-2.18165,0.5,0.25,-1.18482,0.50754,0.27922,-14.8709,1.0217,-201.36047,147.47986,0.08813
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:2705:f708_snr10_QPSK__L04__TT004__S358352__stress_low_snr,f708_snr10_QPSK,708,10,f708_snr10_QPSK__L04,B1_hard_select,True,31.59396,0.32609,0.17174,1,0.54118,-2.66659,0.07775,0.39893,0.12663,230,75,79,0.16667,0.08333,-3.30946,0,0,-4.57415,0.61877,0.21252,-14.81075,1.13781,-31.07592,207.76441,-0.13603
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:2744:f708_snr10_QPSK__L06__TT016__S430736__stress_low_snr,f708_snr10_QPSK,708,10,f708_snr10_QPSK__L06,B1_hard_select,True,27.2249,0.36087,0.18913,1,0.56941,-2.44578,1.20715,0.42839,0.13456,230,83,87,0.16667,0.08333,-1.79923,0,0,-6.69316,0.73274,0.17644,-14.4421,1.42727,-569.7365,-390.89617,1.125
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:2829:f708_snr0_QPSK__L04__TT009__S388512__stress_cfo_phase_jump,f708_snr0_QPSK,708,0,f708_snr0_QPSK__L04,B1_hard_select,True,30.31225,0.44783,0.24783,1,0.68747,-1.62745,3.14404,0.50188,0.16683,230,103,114,0.16667,0.08333,-3.63606,0.5,0.375,2.56541,0.81869,0.10666,-17.36138,0.92937,-728.75979,-132.58404,1.08934
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:2852:f708_snr0_QPSK__L06__TT000__S334224__stress_low_snr,f708_snr0_QPSK,708,0,f708_snr0_QPSK__L06,B1_hard_select,True,71.81959,0.5087,0.27826,1,0.79991,-0.96961,2.79147,0.58385,0.19816,230,117,128,0.16667,0.16667,-0.33186,0,0,-5.27445,0.90789,0.41764,-16.36057,2.02574,-522.91596,-135.98297,1.125
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:2898:f433_snr20_QPSK__L01__TT001__S340256__stress_cfo_phase_jump,f433_snr20_QPSK,433,20,f433_snr20_QPSK__L01,B1_hard_select,True,34.71029,0.18261,0.0913,1,0.33881,-4.70043,-2.98886,0.23609,0.06765,230,42,42,0,0,-5.38742,0,0,-10.14449,0.41746,0.07945,-12.75066,1.71861,-463.41623,-616.07849,-0.01693
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:2905:f433_snr20_QPSK__L01__TT010__S394544__stress_burst_impairment,f433_snr20_QPSK,433,20,f433_snr20_QPSK__L01,B1_hard_select,True,30.98553,0.22609,0.11957,1,0.50465,-2.97013,-0.72134,0.35963,0.08953,230,52,55,0,0,-5.01035,0,0,-6.23347,0.4296,0.09017,-13.00191,0.89604,-209.40075,-212.43601,-0.1013
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:2946:f433_snr20_QPSK__L04__TT005__S364384__stress_cfo_phase_jump,f433_snr20_QPSK,433,20,f433_snr20_QPSK__L04,B1_hard_select,True,31.57737,0.10435,0.05217,1,0.30458,-5.16295,-4.35964,0.17625,0.04947,230,24,24,0,0,-8.23094,0.25,0.125,-4.18151,0.36026,0.14612,-13.16113,0.83725,-209.40076,-517.8628,0.94303
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:2951:f433_snr20_QPSK__L04__TT012__S406608__stress_low_snr,f433_snr20_QPSK,433,20,f433_snr20_QPSK__L04,B1_hard_select,True,51.18731,0.18261,0.09348,1,0.36742,-4.34843,-2.83161,0.23565,0.06881,230,42,43,0,0,-4.34906,0,0,-6.61182,0.36076,0.0498,-12.43084,1.11769,-363.66344,-205.81676,1.125
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:2954:f433_snr20_QPSK__L04__TT016__S430736__stress_low_snr,f433_snr20_QPSK,433,20,f433_snr20_QPSK__L04,B1_hard_select,True,30.75561,0.36087,0.19565,1,0.54466,-2.63874,0.6568,0.41095,0.13351,230,83,90,0,0,-6.09654,0.25,0.125,-1.856,0.63639,0.00879,-12.39961,1.13422,-195.07457,-27.22789,-0.18347
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:2963:f433_snr20_QPSK__L05__TT008__S382480__stress_low_snr,f433_snr20_QPSK,433,20,f433_snr20_QPSK__L05,B1_hard_select,True,30.44169,0.38261,0.21522,1,0.59729,-2.23816,1.55379,0.44104,0.1459,230,88,99,0.16667,0.16667,0.49939,0.25,0.125,-4.33216,0.53224,0.04654,-12.40851,0.99355,-191.12032,-28.27364,-0.93609
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:2974:f433_snr20_QPSK__L06__TT002__S346288__stress_burst_impairment,f433_snr20_QPSK,433,20,f433_snr20_QPSK__L06,B1_hard_select,True,31.35677,0.25652,0.12826,1,0.56661,-2.46716,-0.35591,0.36058,0.09563,230,59,59,0,0,-5.55771,0,0,-4.25565,0.33286,0.02215,-13.11403,0.77245,-190.95352,-79.32592,1.00322
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:2987:f433_snr20_QPSK__L07__TT000__S334224__stress_low_snr,f433_snr20_QPSK,433,20,f433_snr20_QPSK__L07,B1_hard_select,True,82.13056,0.15652,0.08478,1,0.39125,-4.07549,-2.679,0.23605,0.07041,230,36,39,0.16667,0.08333,-3.3814,0,0,-5.97578,0.44815,0.10805,-12.67399,0.72929,-244.19021,-66.34353,0.57761
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:2996:f433_snr20_QPSK__L07__TT012__S406608__stress_low_snr,f433_snr20_QPSK,433,20,f433_snr20_QPSK__L07,B1_hard_select,True,57.53324,0.36087,0.19348,1,0.53521,-2.71477,-0.14993,0.40234,0.12918,230,83,89,0,0,-7.1692,0,0,-3.89426,0.43061,0.179,-12.50065,1.04927,95.1726,273.01928,0.98778
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:2997:f433_snr20_QPSK__L07__TT013__S412640__stress_cfo_phase_jump,f433_snr20_QPSK,433,20,f433_snr20_QPSK__L07,B1_hard_select,True,58.47495,0.06957,0.03478,1,0.23698,-6.25293,-5.40575,0.09257,0.02672,230,16,16,0,0,-8.15933,0,0,-10.80761,0.28371,0.0392,-13.32572,1.12044,-76.16918,-103.36085,0.00791
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:3008:f433_snr10_QPSK__L00__TT008__S382480__stress_low_snr,f433_snr10_QPSK,433,10,f433_snr10_QPSK__L00,B1_hard_select,True,30.0772,0.32174,0.17826,1,0.50058,-3.00524,-0.07584,0.36069,0.11453,230,74,82,0.16667,0.16667,-0.07202,0.25,0.125,-6.30405,0.43321,0.08543,-14.99254,0.75791,-481.73799,-257.3929,0.3324
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:3054:f433_snr10_QPSK__L03__TT009__S388512__stress_cfo_phase_jump,f433_snr10_QPSK,433,10,f433_snr10_QPSK__L03,B1_hard_select,True,25.89582,0.13478,0.06739,1,0.27148,-5.6626,-4.39653,0.16696,0.0484,230,31,31,0,0,-9.36047,0,0,-4.35185,0.20868,0.12835,-16.24602,0.99123,-1135.01467,-527.46803,-0.25506
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:3061:f433_snr10_QPSK__L03__TT018__S442800__stress_burst_impairment,f433_snr10_QPSK,433,10,f433_snr10_QPSK__L03,B1_hard_select,True,31.50269,0.22609,0.11739,1,0.50028,-3.00783,-0.15326,0.33774,0.09272,230,52,54,0.33333,0.16667,0.36065,0,0,-7.48224,0.45551,0.01468,-15.03154,0.88903,-25.05823,10.18503,0.25548
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:3087:f433_snr10_QPSK__L05__TT013__S412640__stress_cfo_phase_jump,f433_snr10_QPSK,433,10,f433_snr10_QPSK__L05,B1_hard_select,True,33.44276,0.0087,0.00435,1,0.15017,-8.2343,-7.98597,0.01672,0.00444,230,2,2,0,0,-10.68252,0,0,-8.37108,0.15119,0.02271,-16.36019,1.53026,-478.08346,109.19165,-0.4375
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:3104:f433_snr10_QPSK__L06__TT016__S430736__stress_low_snr,f433_snr10_QPSK,433,10,f433_snr10_QPSK__L06,B1_hard_select,True,39.26142,0.2913,0.15652,1,0.48907,-3.1063,-0.29387,0.36004,0.11297,230,67,72,0,0,-9.78835,0.25,0.125,-3.14348,0.57916,0.16135,-14.71585,1.15114,-148.34308,61.00201,0.88714
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:3125:f433_snr0_QPSK__L00__TT004__S358352__stress_low_snr,f433_snr0_QPSK,433,0,f433_snr0_QPSK__L00,B1_hard_select,True,59.95321,0.36087,0.2,1,0.59603,-2.24734,1.24347,0.44116,0.1418,230,83,92,0.33333,0.16667,0.64783,0.25,0.125,-1.49847,0.41703,0.03913,-16.31635,0.51927,-220.72278,-87.41285,0.60174
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:3130:f433_snr0_QPSK__L00__TT010__S394544__stress_burst_impairment,f433_snr0_QPSK,433,0,f433_snr0_QPSK__L00,B1_hard_select,True,28.03583,0.56522,0.33696,1,0.9471,-0.23603,5.89566,0.7634,0.2412,230,130,155,0.16667,0.08333,-1.44394,0.25,0.125,-2.11965,0.85853,0.51763,-15.61127,1.4272,-787.47097,-395.2731,-0.9036
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:3164:f433_snr0_QPSK__L02__TT016__S430736__stress_low_snr,f433_snr0_QPSK,433,0,f433_snr0_QPSK__L02,B1_hard_select,True,28.74227,0.36087,0.20217,1,0.61731,-2.09498,1.98646,0.45252,0.14839,230,83,93,0.33333,0.16667,0.45785,0,0,-7.72458,0.61575,0.13702,-16.30063,0.58185,100.86879,289.17873,0.16778
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:3174:f433_snr0_QPSK__L03__TT009__S388512__stress_cfo_phase_jump,f433_snr0_QPSK,433,0,f433_snr0_QPSK__L03,B1_hard_select,True,30.51504,0.36087,0.18261,1,0.55291,-2.57345,-0.65216,0.47022,0.13587,230,83,84,0,0,-10.11373,0,0,-4.62285,0.76134,0.32911,-16.11849,2.05578,-306.2582,-615.79273,-0.02421
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:3182:f433_snr0_QPSK__L04__TT000__S334224__stress_low_snr,f433_snr0_QPSK,433,0,f433_snr0_QPSK__L04,B1_hard_select,True,32.43605,0.4,0.21087,1,0.64357,-1.91406,1.05923,0.53464,0.16424,230,92,97,0,0,-8.45705,0.25,0.125,-1.95367,0.65393,0.85275,-15.96417,0.91676,-181.92991,6.38002,0.0851
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:3191:f433_snr0_QPSK__L04__TT012__S406608__stress_low_snr,f433_snr0_QPSK,433,0,f433_snr0_QPSK__L04,B1_hard_select,True,37.30829,0.36087,0.19783,1,0.60972,-2.14867,1.82588,0.44652,0.14384,230,83,91,0.16667,0.08333,-0.42297,0,0,-9.23795,0.50304,0.02552,-15.85265,1.02975,-230.5524,-77.24246,0.91493
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:3230:f433_snr0_QPSK__L07__TT004__S358352__stress_low_snr,f433_snr0_QPSK,433,0,f433_snr0_QPSK__L07,B1_hard_select,True,58.50169,0.41739,0.22826,1,0.65383,-1.84537,2.73357,0.4957,0.16303,230,96,105,0.16667,0.16667,-2.14173,0.25,0.125,-5.38797,0.83047,0.21382,-15.82238,1.03609,3.31074,141.62068,0.03368
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:3357:f2450_snr20_QPSK__L07__TT013__S412640__stress_cfo_phase_jump,f2450_snr20_QPSK,2450,20,f2450_snr20_QPSK__L07,B2_linear_fusion_plugin,True,32.29867,0.72609,0.48696,1,1.61397,2.07894,14.82526,2.46305,0.44014,230,167,224,0.16667,0.08333,-5.58077,0,0,-4.39736,2.19009,2.745,-5.75916,8.40329,1701.61166,3360.31448,0.10883
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:3363:f2450_snr10_QPSK__L00__TT001__S340256__stress_cfo_phase_jump,f2450_snr10_QPSK,2450,10,f2450_snr10_QPSK__L00,B2_linear_fusion_plugin,True,34.19115,0.12174,0.06087,1,0.25327,-5.96408,-5.36022,0.18962,0.04474,230,28,28,0,0,-14.43316,0,0,-7.17947,0.09878,0.05149,-17.08205,0.76227,-1258.78498,-549.79872,0.90759
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:3381:f2450_snr10_QPSK__L01__TT005__S364384__stress_cfo_phase_jump,f2450_snr10_QPSK,2450,10,f2450_snr10_QPSK__L01,B2_linear_fusion_plugin,True,28.29109,0.09565,0.04783,1,0.22885,-6.40446,-5.58852,0.10998,0.03419,230,22,22,0,0,-14.80504,0,0,-4.3603,0.15431,0.03413,-16.68324,0.68941,-70.01044,478.96802,0.64739
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:3396:f2450_snr10_QPSK__L02__TT005__S364384__stress_cfo_phase_jump,f2450_snr10_QPSK,2450,10,f2450_snr10_QPSK__L02,B2_linear_fusion_plugin,True,32.36034,0.04783,0.02391,1,0.19843,-7.02391,-6.07808,0.05668,0.01708,230,11,11,0,0,-13.22379,0,0,-5.42652,0.1554,0.06195,-16.60965,0.83272,-1327.02556,-352.86193,0.77124
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:3413:f2450_snr10_QPSK__L03__TT008__S382480__stress_low_snr,f2450_snr10_QPSK,2450,10,f2450_snr10_QPSK__L03,B2_linear_fusion_plugin,True,29.83603,0.3,0.15217,1,0.48974,-3.10033,-0.912,0.35869,0.10947,230,69,70,0.66667,0.33333,1.42894,0.5,0.25,-0.82041,0.42429,0.05496,-15.09315,0.54543,-1300.16339,-132.14398,1
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:3421:f2450_snr10_QPSK__L03__TT018__S442800__stress_burst_impairment,f2450_snr10_QPSK,2450,10,f2450_snr10_QPSK__L03,B2_linear_fusion_plugin,True,30.35842,0.36087,0.20217,1,0.57651,-2.39193,1.21199,0.41217,0.1291,230,83,93,0.33333,0.16667,-0.63971,0.5,0.25,2.4345,0.3568,0.06625,-15.50808,0.46025,-1274.71181,-27.34823,1.06677
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:3434:f2450_snr10_QPSK__L04__TT016__S430736__stress_low_snr,f2450_snr10_QPSK,2450,10,f2450_snr10_QPSK__L04,B2_linear_fusion_plugin,True,33.99746,0.32174,0.17174,1,0.50063,-3.00486,-0.49637,0.34336,0.11308,230,74,79,0,0,-8.49403,0,0,-3.44092,0.81634,0.24096,-14.85149,1.09693,-1360.03216,-197.01275,1.125
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:3458:f2450_snr10_QPSK__L06__TT008__S382480__stress_low_snr,f2450_snr10_QPSK,2450,10,f2450_snr10_QPSK__L06,B2_linear_fusion_plugin,True,33.79103,0.36087,0.18913,1,0.60603,-2.17509,0.21621,0.45756,0.14412,230,83,87,0.16667,0.08333,-3.15504,0.25,0.125,-2.12589,0.55337,0.1774,-14.78942,1.30123,-1144.15097,33.86844,-0.03917
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:3468:f2450_snr10_QPSK__L07__TT001__S340256__stress_cfo_phase_jump,f2450_snr10_QPSK,2450,10,f2450_snr10_QPSK__L07,B2_linear_fusion_plugin,True,28.75561,0.76087,0.51087,1,1.7443,2.4162,12.17109,2.38673,0.46517,230,175,235,0,0,-10.14679,0.5,0.25,-0.36007,1.99122,2.1013,-10.52072,6.56112,1592.81108,2834.34807,0.71042
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:3484:f2450_snr0_QPSK__L00__TT002__S346288__stress_burst_impairment,f2450_snr0_QPSK,2450,0,f2450_snr0_QPSK__L00,B2_linear_fusion_plugin,True,30.86769,0.76522,0.49565,1,1.16052,0.64653,23.36055,0.78964,0.28623,230,176,228,0.5,0.25,3.6924,0.25,0.125,-0.79625,0.65528,0.902,-16.50567,0.79861,333.84321,1448.50604,0.0113
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:3505:f2450_snr0_QPSK__L01__TT010__S394544__stress_burst_impairment,f2450_snr0_QPSK,2450,0,f2450_snr0_QPSK__L01,B2_linear_fusion_plugin,True,32.24587,0.36087,0.19348,1,0.57018,-2.43988,0.89579,0.41333,0.12723,230,83,89,0.16667,0.08333,-4.00153,0.25,0.125,-2.0736,0.69181,0.05763,-16.513,0.97634,-928.06267,-35.39597,-0.10962
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:3515:f2450_snr0_QPSK__L02__TT004__S358352__stress_low_snr,f2450_snr0_QPSK,2450,0,f2450_snr0_QPSK__L02,B2_linear_fusion_plugin,True,29.90841,0.36087,0.19348,1,0.57187,-2.42703,1.03129,0.4151,0.13248,230,83,89,0.33333,0.16667,0.84609,0,0,-13.80238,0.66342,0.01327,-16.57917,0.77742,-1177.71096,75.93467,0.14602
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:3620:f708_snr20_QPSK__L01__TT004__S358352__stress_low_snr,f708_snr20_QPSK,708,20,f708_snr20_QPSK__L01,B2_linear_fusion_plugin,True,40.86636,0.15217,0.08043,1,0.36039,-4.43222,-3.16765,0.22181,0.06309,230,35,37,0.16667,0.08333,-1.68485,0.25,0.125,-4.15116,0.44485,0.02537,-12.52755,0.66927,-341.30902,34.69745,0.4752
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:3683:f708_snr20_QPSK__L05__TT008__S382480__stress_low_snr,f708_snr20_QPSK,708,20,f708_snr20_QPSK__L05,B2_linear_fusion_plugin,True,47.73606,0.20435,0.1087,1,0.40247,-3.95264,-2.28679,0.27217,0.07948,230,47,50,0.16667,0.16667,-1.33319,0.25,0.125,-0.02544,0.63074,0.02048,-11.80831,1.15313,-348.03607,22.9704,-0.02555
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:3699:f708_snr20_QPSK__L06__TT009__S388512__stress_cfo_phase_jump,f708_snr20_QPSK,708,20,f708_snr20_QPSK__L06,B2_linear_fusion_plugin,True,32.85147,0.32609,0.16304,1,0.50699,-2.94999,-0.03607,0.44068,0.12992,230,75,75,0.16667,0.08333,-4.76016,0,0,-12.24085,0.31995,0.23519,-12.93664,0.93851,-192.77638,-377.09719,-0.70401
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:3731:f708_snr10_QPSK__L00__TT012__S406608__stress_low_snr,f708_snr10_QPSK,708,10,f708_snr10_QPSK__L00,B2_linear_fusion_plugin,True,32.41159,0.36087,0.19348,1,0.59287,-2.27043,1.48944,0.43778,0.14026,230,83,89,0.5,0.25,0.92017,0,0,-5.78406,0.63875,0.0454,-14.93853,0.83735,-363.85242,-100.01208,-0.8153
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:3785:f708_snr10_QPSK__L04__TT004__S358352__stress_low_snr,f708_snr10_QPSK,708,10,f708_snr10_QPSK__L04,B2_linear_fusion_plugin,True,31.56294,0.33043,0.17391,1,0.54469,-2.63853,0.10987,0.3998,0.12725,230,76,80,0.16667,0.08333,-3.30882,0,0,-4.81974,0.60345,0.20177,-14.83598,1.07623,-59.46714,179.37319,-0.13603
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:3856:f708_snr0_QPSK__L00__TT018__S442800__stress_burst_impairment,f708_snr0_QPSK,708,0,f708_snr0_QPSK__L00,B2_linear_fusion_plugin,True,59.41692,0.67826,0.41957,1,1.00647,0.02801,12.1544,0.70129,0.25082,230,156,193,0.66667,0.33333,11.06508,0,0,-10.52893,1.14747,0.93312,-16.92729,1.13605,-1252.60923,-717.85996,-1.125
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:3900:f708_snr0_QPSK__L03__TT017__S436768__stress_cfo_phase_jump,f708_snr0_QPSK,708,0,f708_snr0_QPSK__L03,B2_linear_fusion_plugin,True,32.2437,0.46522,0.28043,1,0.75458,-1.22296,4.76354,0.56577,0.19268,230,107,129,0.66667,0.33333,1.25158,0.25,0.125,-2.79493,0.9421,0.28931,-17.30553,0.73462,-685.2845,360.07653,-0.11625
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:3922:f708_snr0_QPSK__L05__TT006__S370416__stress_burst_impairment,f708_snr0_QPSK,708,0,f708_snr0_QPSK__L05,B2_linear_fusion_plugin,True,33.67812,0.57391,0.32391,1,0.87868,-0.56169,4.81117,0.63073,0.21965,230,132,149,0.5,0.41667,14.36003,0.75,0.375,5.68078,0.83567,0.63628,-17.69936,0.5546,-351.2865,-490.7299,0.64234
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:3932:f708_snr0_QPSK__L06__TT000__S334224__stress_low_snr,f708_snr0_QPSK,708,0,f708_snr0_QPSK__L06,B2_linear_fusion_plugin,True,71.80265,0.5087,0.27826,1,0.7925,-1.01002,2.89128,0.57837,0.19647,230,117,128,0.16667,0.16667,-0.33382,0,0,-5.17784,0.90742,0.38532,-16.36042,2.0151,-566.17764,-179.24466,1.125
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:4034:f433_snr20_QPSK__L04__TT016__S430736__stress_low_snr,f433_snr20_QPSK,433,20,f433_snr20_QPSK__L04,B2_linear_fusion_plugin,True,30.68709,0.36087,0.19565,1,0.54585,-2.62931,0.67257,0.40677,0.13269,230,83,90,0,0,-6.09631,0.25,0.125,-1.75413,0.61978,0.00878,-12.43157,1.06514,-204.52106,-36.67438,-0.18347
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:4043:f433_snr20_QPSK__L05__TT008__S382480__stress_low_snr,f433_snr20_QPSK,433,20,f433_snr20_QPSK__L05,B2_linear_fusion_plugin,True,30.40318,0.3913,0.22174,1,0.59885,-2.22681,1.67233,0.44423,0.14776,230,90,102,0.16667,0.16667,0.48721,0.25,0.125,-4.00123,0.57901,0.05167,-12.32374,1.0539,-297.89074,-135.04406,-0.93609
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:4061:f433_snr20_QPSK__L06__TT012__S406608__stress_low_snr,f433_snr20_QPSK,433,20,f433_snr20_QPSK__L06,B2_linear_fusion_plugin,True,30.64132,0.3087,0.16304,1,0.52666,-2.78471,0.27215,0.37838,0.12045,230,71,75,0.5,0.25,0.2363,0,0,-2.41655,0.33602,0.07206,-12.78731,0.44812,-72.02228,100.8244,1.125
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:4084:f433_snr10_QPSK__L00__TT002__S346288__stress_burst_impairment,f433_snr10_QPSK,433,10,f433_snr10_QPSK__L00,B2_linear_fusion_plugin,True,31.88785,0.24783,0.13261,1,0.4952,-3.05216,-0.26769,0.33603,0.09854,230,57,61,0,0,-4.16864,0.5,0.25,2.04117,0.40357,0.02538,-15.1558,0.45138,-378.4586,-13.8737,0.29309
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:4210:f433_snr0_QPSK__L00__TT010__S394544__stress_burst_impairment,f433_snr0_QPSK,433,0,f433_snr0_QPSK__L00,B2_linear_fusion_plugin,True,28.02907,0.56522,0.33261,1,0.93449,-0.29425,5.88733,0.7323,0.23651,230,130,153,0.16667,0.08333,-1.44556,0.25,0.125,-2.06066,0.83873,0.47261,-15.66784,1.33604,-768.83981,-376.64194,-0.9036
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:4222:f433_snr0_QPSK__L01__TT006__S370416__stress_burst_impairment,f433_snr0_QPSK,433,0,f433_snr0_QPSK__L01,B2_linear_fusion_plugin,True,28.23263,0.41304,0.23043,1,0.65299,-1.8509,2.72946,0.49892,0.15942,230,95,106,0.16667,0.16667,-2.53129,0.25,0.125,-3.47097,0.81956,0.24222,-15.55543,1.36086,-261.07807,270.54123,-0.76914
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:4234:f433_snr0_QPSK__L02__TT002__S346288__stress_burst_impairment,f433_snr0_QPSK,433,0,f433_snr0_QPSK__L02,B2_linear_fusion_plugin,True,31.41897,0.36087,0.19783,1,0.5903,-2.28925,1.56393,0.42324,0.13337,230,83,91,0.16667,0.08333,-4.40828,0.25,0.125,-0.01936,0.58646,0.00778,-16.25852,0.83192,-376.19838,50.2732,0.30027
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:4252:f433_snr0_QPSK__L03__TT006__S370416__stress_burst_impairment,f433_snr0_QPSK,433,0,f433_snr0_QPSK__L03,B2_linear_fusion_plugin,True,28.90619,0.55217,0.32174,1,0.79586,-0.99161,3.75723,0.58051,0.1981,230,127,148,0.66667,0.33333,2.80063,0.5,0.25,2.10652,0.89285,0.36073,-15.86069,0.88441,477.96599,269.90871,-0.66796
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:4255:f433_snr0_QPSK__L03__TT010__S394544__stress_burst_impairment,f433_snr0_QPSK,433,0,f433_snr0_QPSK__L03,B2_linear_fusion_plugin,True,32.53014,0.36087,0.2087,1,0.66862,-1.74821,2.94398,0.49207,0.15178,230,83,96,0.33333,0.25,1.48544,0,0,-5.04549,0.56073,0.01746,-16.14666,0.87948,418.39139,119.70061,-0.06638
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:4258:f433_snr0_QPSK__L03__TT014__S418672__stress_burst_impairment,f433_snr0_QPSK,433,0,f433_snr0_QPSK__L03,B2_linear_fusion_plugin,True,64.62636,0.36087,0.20652,1,0.66002,-1.80441,1.97498,0.47418,0.14849,230,83,95,0.33333,0.16667,-3.50614,0,0,-2.87405,0.64165,0.14227,-16.21656,0.94696,-133.10543,-103.05531,1.125
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:4271:f433_snr0_QPSK__L04__TT012__S406608__stress_low_snr,f433_snr0_QPSK,433,0,f433_snr0_QPSK__L04,B2_linear_fusion_plugin,True,37.26529,0.36087,0.19783,1,0.61011,-2.14592,1.828,0.44679,0.1439,230,83,91,0.16667,0.08333,-0.42313,0,0,-9.26128,0.50104,0.03061,-15.85501,1.01786,-232.96992,-79.65999,0.91493
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:4300:f433_snr0_QPSK__L06__TT010__S394544__stress_burst_impairment,f433_snr0_QPSK,433,0,f433_snr0_QPSK__L06,B2_linear_fusion_plugin,True,29.17236,0.36087,0.19565,1,0.62805,-2.02007,2.16898,0.46529,0.14202,230,83,90,0.5,0.25,-0.7672,0,0,-4.4386,0.6051,0.09306,-16.11793,0.8953,-379.72747,-283.89343,0.79018
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:4312:f433_snr0_QPSK__L07__TT006__S370416__stress_burst_impairment,f433_snr0_QPSK,433,0,f433_snr0_QPSK__L07,B2_linear_fusion_plugin,True,25.62478,0.36087,0.19783,1,0.6592,-1.8098,1.51189,0.46325,0.14589,230,83,91,0.33333,0.16667,-2.25314,0,0,-1.26057,0.75376,0.81426,-15.79277,0.54017,-322.40128,-166.32405,0.77397
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:4321:f433_snr0_QPSK__L07__TT018__S442800__stress_burst_impairment,f433_snr0_QPSK,433,0,f433_snr0_QPSK__L07,B2_linear_fusion_plugin,True,37.18364,0.36087,0.19348,1,0.60676,-2.16981,1.86199,0.44732,0.1375,230,83,89,0.33333,0.16667,-3.4407,0.25,0.125,-3.03299,0.49115,0.13842,-16.44898,0.61116,-168.27735,-91.87597,1.02923
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:4349:f2450_snr20_QPSK__L01__TT016__S430736__stress_low_snr,f2450_snr20_QPSK,2450,20,f2450_snr20_QPSK__L01,B3_mixture_likelihood,True,47.21054,0.33913,0.18478,1,0.49205,-3.07988,-0.94733,0.37358,0.12301,230,78,85,0,0,-8.52984,0.25,0.125,-3.22497,0.57457,0.09908,-12.45097,0.99353,-1198.73609,41.50378,1.00539
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:4406:f2450_snr20_QPSK__L05__TT012__S406608__stress_low_snr,f2450_snr20_QPSK,2450,20,f2450_snr20_QPSK__L05,B3_mixture_likelihood,True,38.29055,0.36087,0.20217,1,0.58245,-2.34741,1.27128,0.45517,0.14561,230,83,93,0.16667,0.16667,-2.00329,0.25,0.125,-4.807,0.7882,0.0312,-11.83075,1.39992,-1322.39285,-112.15298,-0.90979
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:4407:f2450_snr20_QPSK__L05__TT013__S412640__stress_cfo_phase_jump,f2450_snr20_QPSK,2450,20,f2450_snr20_QPSK__L05,B3_mixture_likelihood,True,32.36062,0.15652,0.07826,1,0.21835,-6.60851,-6.1116,0.17209,0.05459,230,36,36,0,0,-11.44995,0,0,-11.90919,0.22362,0.15255,-13.53612,0.63629,-197.09093,506.408,0.74717
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:4412:f2450_snr20_QPSK__L06__TT000__S334224__stress_low_snr,f2450_snr20_QPSK,2450,20,f2450_snr20_QPSK__L06,B3_mixture_likelihood,True,37.36039,0.24783,0.13261,1,0.37772,-4.22835,-2.39118,0.30643,0.09443,230,57,61,0,0,-16.46177,0,0,-14.23174,0.59239,0.20581,-11.50966,1.90215,-1325.32438,-110.08451,0.09889
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:4495:f2450_snr10_QPSK__L03__TT010__S394544__stress_burst_impairment,f2450_snr10_QPSK,2450,10,f2450_snr10_QPSK__L03,B3_mixture_likelihood,True,35.51221,0.36087,0.18913,1,0.6274,-2.02452,1.13278,0.49838,0.15685,230,83,87,0.83333,0.41667,-7.00856,0.25,0.125,-3.75829,1.53828,1.16578,-14.10352,0.91001,-1225.49341,-442.16264,0.04656
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:4586:f2450_snr0_QPSK__L01__TT012__S406608__stress_low_snr,f2450_snr0_QPSK,2450,0,f2450_snr0_QPSK__L01,B3_mixture_likelihood,True,33.63127,0.76522,0.48913,1,1.06628,0.27872,18.13268,0.72979,0.26657,230,176,225,0.16667,0.08333,-4.36072,0.75,0.375,10.56635,1.55384,1.16416,-16.20158,0.73316,710.22356,1938.86919,1.07176
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:4596:f2450_snr0_QPSK__L02__TT005__S364384__stress_cfo_phase_jump,f2450_snr0_QPSK,2450,0,f2450_snr0_QPSK__L02,B3_mixture_likelihood,True,34.08779,0.28261,0.1413,1,0.43174,-3.64782,-1.911,0.36051,0.10793,230,65,65,0,0,-13.18349,0.25,0.125,-1.0198,0.36126,0.02007,-17.45794,0.56739,-1.92509,642.9137,0.45241
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:4633:f2450_snr0_QPSK__L04__TT014__S418672__stress_burst_impairment,f2450_snr0_QPSK,2450,0,f2450_snr0_QPSK__L04,B3_mixture_likelihood,True,35.33924,0.36087,0.19348,1,0.55852,-2.5296,0.83084,0.44405,0.13963,230,83,89,0.16667,0.08333,-2.39884,0,0,-17.60609,0.58359,0.12398,-16.69527,0.88598,-745.05142,102.31686,0.10404
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:4637:f2450_snr0_QPSK__L05__TT000__S334224__stress_low_snr,f2450_snr0_QPSK,2450,0,f2450_snr0_QPSK__L05,B3_mixture_likelihood,True,32.75253,0.36087,0.22174,1,0.61238,-2.12982,1.85881,0.4677,0.15309,230,83,102,0.33333,0.16667,-2.29972,0.25,0.125,-2.17524,0.51869,0.13754,-16.44046,0.74544,-1272.40242,1.24321,-0.03419
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:4650:f2450_snr0_QPSK__L05__TT017__S436768__stress_cfo_phase_jump,f2450_snr0_QPSK,2450,0,f2450_snr0_QPSK__L05,B3_mixture_likelihood,True,31.98762,0.37826,0.20652,1,0.59009,-2.29085,0.69734,0.47794,0.14752,230,87,95,0.5,0.25,-0.15598,0,0,-14.61393,0.3382,0.10171,-17.57905,0.47504,-1162.83608,-521.86365,0.01377
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:4665:f2450_snr0_QPSK__L06__TT017__S436768__stress_cfo_phase_jump,f2450_snr0_QPSK,2450,0,f2450_snr0_QPSK__L06,B3_mixture_likelihood,True,34.1758,0.16087,0.08261,1,0.36057,-4.43004,-3.59819,0.32328,0.09014,230,37,38,0.5,0.25,1.15952,0.25,0.125,-3.48353,1.06606,1.07413,-16.392,0.8755,-1085.08915,564.67931,0.08891
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:4724:f708_snr20_QPSK__L02__TT016__S430736__stress_low_snr,f708_snr20_QPSK,708,20,f708_snr20_QPSK__L02,B3_mixture_likelihood,True,44.85872,0.23913,0.13043,1,0.36002,-4.43672,-2.5811,0.29095,0.09001,230,55,60,0.16667,0.08333,-6.12854,0.5,0.25,3.02575,0.41024,0.03814,-12.21243,0.99711,-363.77311,27.23336,0.14932
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:4730:f708_snr20_QPSK__L03__TT004__S358352__stress_low_snr,f708_snr20_QPSK,708,20,f708_snr20_QPSK__L03,B3_mixture_likelihood,True,38.52318,0.28696,0.15652,1,0.42884,-3.677,-2.45291,0.33609,0.10721,230,66,72,0.16667,0.16667,-0.66743,0,0,-13.68556,0.44582,0.12941,-12.38063,0.82482,-369.2624,1.74407,1.125
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:4763:f708_snr20_QPSK__L05__TT008__S382480__stress_low_snr,f708_snr20_QPSK,708,20,f708_snr20_QPSK__L05,B3_mixture_likelihood,True,48.88989,0.1913,0.1,1,0.31452,-5.02355,-3.81686,0.26969,0.07863,230,44,46,0.16667,0.16667,-1.64253,0.25,0.125,-0.2564,0.63074,0.02048,-11.80831,1.15313,-360.65285,10.35362,-0.02555
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:4771:f708_snr20_QPSK__L05__TT018__S442800__stress_burst_impairment,f708_snr20_QPSK,708,20,f708_snr20_QPSK__L05,B3_mixture_likelihood,True,31.13861,0.21304,0.11304,1,0.33367,-4.76687,-3.07614,0.33602,0.08342,230,49,52,0,0,-20.09003,0.25,0.125,-6.00187,0.41209,0.08499,-12.75915,0.83822,-641.82342,-114.50897,0.92472
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:4799:f708_snr20_QPSK__L07__TT016__S430736__stress_low_snr,f708_snr20_QPSK,708,20,f708_snr20_QPSK__L07,B3_mixture_likelihood,True,33.22837,0.17826,0.0913,1,0.25295,-5.9696,-5.10025,0.20942,0.06324,230,41,42,0.33333,0.16667,-3.33603,0,0,-11.63897,0.52201,0.33899,-12.31728,0.91416,-337.99638,43.01009,0.50944
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:4865:f708_snr10_QPSK__L04__TT004__S358352__stress_low_snr,f708_snr10_QPSK,708,10,f708_snr10_QPSK__L04,B3_mixture_likelihood,True,32.79063,0.30435,0.16304,1,0.47827,-3.20328,-0.70194,0.37679,0.11957,230,70,75,0.33333,0.16667,-4.519,0,0,-12.51508,0.60345,0.20177,-14.83598,1.07623,-159.62262,79.21771,-0.13603
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:4897:f708_snr10_QPSK__L06__TT006__S370416__stress_burst_impairment,f708_snr10_QPSK,708,10,f708_snr10_QPSK__L06,B3_mixture_likelihood,True,54.0869,0.55652,0.31087,1,0.87818,-0.56416,0.55509,0.63122,0.21955,230,128,143,1,0.58333,-4.76578,0.25,0.125,2.13607,0.97108,1.0776,-14.60408,0.45388,-223.38242,512.59729,0.44605
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:4916:f708_snr10_QPSK__L07__TT012__S406608__stress_low_snr,f708_snr10_QPSK,708,10,f708_snr10_QPSK__L07,B3_mixture_likelihood,True,51.36097,0.34783,0.18696,1,0.52483,-2.79977,-1.46422,0.41135,0.13121,230,80,86,0.16667,0.08333,-7.97981,0,0,-11.46092,0.51402,0.49691,-14.93085,1.22783,-355.86365,187.97668,0.59146
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:4936:f708_snr0_QPSK__L00__TT018__S442800__stress_burst_impairment,f708_snr0_QPSK,708,0,f708_snr0_QPSK__L00,B3_mixture_likelihood,True,60.64341,0.70435,0.43696,1,1.02927,0.12528,12.35415,0.71996,0.25732,230,162,201,0.5,0.25,9.96616,0,0,-8.1583,1.14747,0.93312,-16.92729,1.13605,-1394.87592,-860.12666,-1.125
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:4980:f708_snr0_QPSK__L03__TT017__S436768__stress_cfo_phase_jump,f708_snr0_QPSK,708,0,f708_snr0_QPSK__L03,B3_mixture_likelihood,True,33.25883,0.47391,0.28261,1,0.76099,-1.18623,4.7519,0.55652,0.19025,230,109,130,0.66667,0.33333,1.46702,0.25,0.125,-2.45518,0.9421,0.28931,-17.30553,0.73462,-684.371,360.99003,-0.11625
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:5012:f708_snr0_QPSK__L06__TT000__S334224__stress_low_snr,f708_snr0_QPSK,708,0,f708_snr0_QPSK__L06,B3_mixture_likelihood,True,72.86213,0.52609,0.28913,1,0.78151,-1.07066,2.85817,0.57525,0.19538,230,121,133,0.16667,0.16667,-1.14832,0,0,-5.95829,0.90742,0.38532,-16.36042,2.0151,-603.28251,-216.34953,1.125
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:5013:f708_snr0_QPSK__L06__TT001__S340256__stress_cfo_phase_jump,f708_snr0_QPSK,708,0,f708_snr0_QPSK__L06,B3_mixture_likelihood,True,61.71697,0.6087,0.36087,1,0.87135,-0.59809,7.66316,0.6227,0.21784,230,140,166,0.33333,0.16667,1.14425,0.5,0.25,1.16014,0.62907,0.4147,-17.2429,0.82328,-1099.18069,-939.76542,1.125
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:5095:f433_snr20_QPSK__L03__TT010__S394544__stress_burst_impairment,f433_snr20_QPSK,433,20,f433_snr20_QPSK__L03,B3_mixture_likelihood,True,33.09348,0.22174,0.12609,1,0.37449,-4.26554,-2.28373,0.32575,0.09362,230,51,58,0.16667,0.08333,-4.22841,0.25,0.125,-1.3607,0.37913,0.1056,-12.91681,0.61417,-292.08527,-256.97098,0.21592
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:5100:f433_snr20_QPSK__L03__TT017__S436768__stress_cfo_phase_jump,f433_snr20_QPSK,433,20,f433_snr20_QPSK__L03,B3_mixture_likelihood,True,28.58542,0.15652,0.07826,1,0.23608,-6.2694,-5.12932,0.19584,0.05902,230,36,36,0,0,-42.33362,0.25,0.125,-7.67651,0.37675,0.13814,-13.22009,1.20328,-143.51111,322.5886,-0.28642
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:5123:f433_snr20_QPSK__L05__TT008__S382480__stress_low_snr,f433_snr20_QPSK,433,20,f433_snr20_QPSK__L05,B3_mixture_likelihood,True,31.66242,0.38261,0.21739,1,0.59859,-2.22869,1.62107,0.44964,0.14965,230,88,100,0.16667,0.16667,-0.75581,0.25,0.125,-5.06416,0.57901,0.05167,-12.32374,1.0539,-355.35371,-192.50703,-0.93609
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:5141:f433_snr20_QPSK__L06__TT012__S406608__stress_low_snr,f433_snr20_QPSK,433,20,f433_snr20_QPSK__L06,B3_mixture_likelihood,True,32.07392,0.26957,0.14348,1,0.45123,-3.45601,-1.04249,0.36125,0.11281,230,62,66,0.5,0.25,0.08313,0.25,0.125,-5.54492,0.33602,0.07206,-12.78731,0.44812,-104.1143,68.73238,1.125
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:5165:f433_snr10_QPSK__L00__TT004__S358352__stress_low_snr,f433_snr10_QPSK,433,10,f433_snr10_QPSK__L00,B3_mixture_likelihood,True,31.87649,0.27826,0.15,1,0.44817,-3.48561,-0.96697,0.3605,0.11204,230,64,69,0,0,-10.9084,0.5,0.25,1.93736,0.46307,0.07888,-14.80833,0.78206,-119.372,114.97309,0.95425
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:5255:f433_snr10_QPSK__L06__TT004__S358352__stress_low_snr,f433_snr10_QPSK,433,10,f433_snr10_QPSK__L06,B3_mixture_likelihood,True,36.8078,0.36087,0.19348,1,0.56926,-2.4469,0.87203,0.45043,0.14231,230,83,89,0,0,-12.93028,0.5,0.375,1.55014,0.76862,0.0489,-14.76127,0.95361,-498.2272,-283.88211,0.9505
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:5270:f433_snr10_QPSK__L07__TT004__S358352__stress_low_snr,f433_snr10_QPSK,433,10,f433_snr10_QPSK__L07,B3_mixture_likelihood,True,43.93572,0.36087,0.19565,1,0.50391,-2.97647,-1.59047,0.40198,0.12598,230,83,90,0.33333,0.16667,-4.22822,0.5,0.25,0.34902,0.85532,0.06329,-14.81429,0.74416,-3.50608,220.83902,-0.47536
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:5290:f433_snr0_QPSK__L00__TT010__S394544__stress_burst_impairment,f433_snr0_QPSK,433,0,f433_snr0_QPSK__L00,B3_mixture_likelihood,True,29.08895,0.57826,0.34783,1,0.94306,-0.2546,5.56184,0.71776,0.23577,230,133,160,0.16667,0.08333,-3.4291,0.25,0.125,-2.87601,0.83873,0.47261,-15.66784,1.33604,-787.52984,-395.33198,-0.9036
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:5302:f433_snr0_QPSK__L01__TT006__S370416__stress_burst_impairment,f433_snr0_QPSK,433,0,f433_snr0_QPSK__L01,B3_mixture_likelihood,True,29.45537,0.41304,0.23043,1,0.63897,-1.94518,2.32586,0.49456,0.15974,230,95,106,0.16667,0.16667,-3.27203,0.25,0.125,-4.32883,0.81956,0.24222,-15.55543,1.36086,-238.55453,293.06477,-0.76914
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:5332:f433_snr0_QPSK__L03__TT006__S370416__stress_burst_impairment,f433_snr0_QPSK,433,0,f433_snr0_QPSK__L03,B3_mixture_likelihood,True,30.04319,0.51304,0.30217,1,0.74542,-1.27596,3.42305,0.55521,0.18636,230,118,139,0.5,0.25,2.7164,0.25,0.125,-0.00429,0.89285,0.36073,-15.86069,0.88441,170.6058,-37.45147,-0.66796
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:5345:f433_snr0_QPSK__L04__TT004__S358352__stress_low_snr,f433_snr0_QPSK,433,0,f433_snr0_QPSK__L04,B3_mixture_likelihood,True,27.43623,0.65217,0.4,1,0.89078,-0.50229,7.6348,0.63602,0.2227,230,150,184,0,0,-9.00617,1,0.5,7.50905,1.19879,0.78992,-15.28713,1.25415,-129.66513,-21.3552,-1.09415
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:5351:f433_snr0_QPSK__L04__TT012__S406608__stress_low_snr,f433_snr0_QPSK,433,0,f433_snr0_QPSK__L04,B3_mixture_likelihood,True,38.3626,0.36087,0.19783,1,0.57536,-2.40057,1.31108,0.44652,0.14384,230,83,91,0.16667,0.08333,-1.59441,0,0,-14.71071,0.50104,0.03061,-15.85501,1.01786,-230.60405,-77.29412,0.91493
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:5404:f2450_snr20_QPSK__L00__TT002__S346288__stress_burst_impairment,f2450_snr20_QPSK,2450,20,f2450_snr20_QPSK__L00,C2_attn_weight_resid_net,True,34.53146,0.1913,0.1,1,0.35748,-4.46753,-2.82841,0.36083,0.08937,230,44,46,0,0,-7.56453,0.25,0.125,-5.91029,0.32553,0.05092,-12.87282,0.96504,-886.86702,79.93761,-0.02353
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:5508:f2450_snr20_QPSK__L07__TT001__S340256__stress_cfo_phase_jump,f2450_snr20_QPSK,2450,20,f2450_snr20_QPSK__L07,C2_attn_weight_resid_net,True,33.2606,0.08261,0.0413,1,0.13602,-8.66385,-8.08364,0.1213,0.03401,230,19,19,0,0,-8.16763,0,0,-10.77576,0.27333,0.06321,-13.08595,1.0999,-1320.08543,53.00301,1.125
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:5522:f2450_snr10_QPSK__L00__TT000__S334224__stress_low_snr,f2450_snr10_QPSK,2450,10,f2450_snr10_QPSK__L00,C2_attn_weight_resid_net,True,42.80683,0.67826,0.42174,1,1.08932,0.37155,5.62068,0.78544,0.27233,230,156,194,0,0,-7.44972,0.5,0.25,4.74819,1.14297,0.92462,-14.36077,1.34544,-1819.20139,-651.18198,-0.97709
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:5524:f2450_snr10_QPSK__L00__TT002__S346288__stress_burst_impairment,f2450_snr10_QPSK,2450,10,f2450_snr10_QPSK__L00,C2_attn_weight_resid_net,True,33.26233,0.36087,0.1913,1,0.51784,-2.85806,-0.65833,0.4112,0.12946,230,83,88,0.33333,0.16667,-1.34983,0,0,-9.01116,0.3697,0.1046,-15.43976,0.68149,-1184.85111,-345.1255,1.05046
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:5558:f2450_snr10_QPSK__L02__TT008__S382480__stress_low_snr,f2450_snr10_QPSK,2450,10,f2450_snr10_QPSK__L02,C2_attn_weight_resid_net,True,34.14629,0.27826,0.14783,1,0.45202,-3.44841,-1.43751,0.36821,0.11301,230,64,68,0.33333,0.16667,-0.03473,0.25,0.125,-2.16486,0.54897,0.13808,-14.9042,0.9365,-926.03602,256.98339,0.96576
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:5561:f2450_snr10_QPSK__L02__TT012__S406608__stress_low_snr,f2450_snr10_QPSK,2450,10,f2450_snr10_QPSK__L02,C2_attn_weight_resid_net,True,36.34352,0.22174,0.11739,1,0.35208,-4.53358,-2.86152,0.28882,0.08802,230,51,54,0,0,-12.20312,0.25,0.125,-6.07648,0.46861,0.02939,-15.35504,0.57306,-1256.37967,-53.36026,1.09293
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:5587:f2450_snr10_QPSK__L04__TT006__S370416__stress_burst_impairment,f2450_snr10_QPSK,2450,10,f2450_snr10_QPSK__L04,C2_attn_weight_resid_net,True,32.5554,0.2087,0.11304,1,0.36091,-4.42602,-2.57965,0.32072,0.09023,230,48,52,0,0,-10.11052,0,0,-9.83852,0.6441,0.03216,-15.13482,0.90543,-990.15532,-150.28588,1.125
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:5607:f2450_snr10_QPSK__L05__TT013__S412640__stress_cfo_phase_jump,f2450_snr10_QPSK,2450,10,f2450_snr10_QPSK__L05,C2_attn_weight_resid_net,True,38.9725,0,0,0,0.00121,-29.183,-29.25704,0.00222,0.00030174,230,0,0,0,0,-44.36007,0,0,-70.75232,0.12893,0.04682,-16.93487,1.07267,-671.61877,12.25213,0.61066
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:5627:f2450_snr10_QPSK__L07__TT000__S334224__stress_low_snr,f2450_snr10_QPSK,2450,10,f2450_snr10_QPSK__L07,C2_attn_weight_resid_net,True,35.31637,0.36087,0.2,1,0.53367,-2.72726,-1.23865,0.41422,0.13342,230,83,92,0.16667,0.08333,-1.87878,0.25,0.125,-2.70088,0.50184,0.19521,-15.13446,0.8537,-1299.86395,-91.84454,1
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:5631:f2450_snr10_QPSK__L07__TT005__S364384__stress_cfo_phase_jump,f2450_snr10_QPSK,2450,10,f2450_snr10_QPSK__L07,C2_attn_weight_resid_net,True,33.02067,0.01304,0.00652,1,0.08557,-10.6766,-11.31011,0.11082,0.02139,230,3,3,0,0,-26.76975,0.25,0.125,-7.74139,0.10534,0.02444,-16.33601,0.17866,-206.74514,412.29248,0.82035
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:5644:f2450_snr0_QPSK__L00__TT002__S346288__stress_burst_impairment,f2450_snr0_QPSK,2450,0,f2450_snr0_QPSK__L00,C2_attn_weight_resid_net,True,32.92016,0.36087,0.21739,1,0.61544,-2.10817,1.60348,0.47902,0.15386,230,83,100,0.66667,0.33333,4.0505,0.5,0.25,1.74072,0.55821,0.08934,-16.6268,0.68086,-1225.17279,-110.50996,0.0113
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:5665:f2450_snr0_QPSK__L01__TT010__S394544__stress_burst_impairment,f2450_snr0_QPSK,2450,0,f2450_snr0_QPSK__L01,C2_attn_weight_resid_net,True,34.25365,0.36087,0.19348,1,0.50853,-2.93685,-0.10036,0.41175,0.12713,230,83,89,0.16667,0.08333,-4.94328,0.25,0.125,-3.56366,0.66849,0.03449,-16.54815,1.01358,-944.34287,-51.67617,-0.10962
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:5666:f2450_snr0_QPSK__L01__TT012__S406608__stress_low_snr,f2450_snr0_QPSK,2450,0,f2450_snr0_QPSK__L01,C2_attn_weight_resid_net,True,34.45244,0.65217,0.36087,1,0.8645,-0.63233,4.05826,0.61956,0.21613,230,150,166,0.16667,0.08333,-4.08074,0.75,0.625,6.69122,1.21052,0.53109,-16.36148,0.64193,-901.75095,326.89468,1.07176
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:5672:f2450_snr0_QPSK__L02__TT000__S334224__stress_low_snr,f2450_snr0_QPSK,2450,0,f2450_snr0_QPSK__L02,C2_attn_weight_resid_net,True,27.53679,0.47826,0.26957,1,0.73609,-1.33069,3.32641,0.54809,0.18402,230,110,124,0.16667,0.08333,-2.45033,0.5,0.25,2.11444,0.92391,0.17022,-16.18513,1.13523,-1475.84699,-147.20136,1.125
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:5693:f2450_snr0_QPSK__L03__TT008__S382480__stress_low_snr,f2450_snr0_QPSK,2450,0,f2450_snr0_QPSK__L03,C2_attn_weight_resid_net,True,31.87136,0.36957,0.21087,1,0.63067,-2.00195,1.83079,0.48659,0.15767,230,85,97,0.66667,0.33333,4.53289,0.25,0.125,-2.0109,0.74025,0.02213,-16.37902,0.50091,-1360.58412,38.06151,0.06789
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:5717:f2450_snr0_QPSK__L05__TT000__S334224__stress_low_snr,f2450_snr0_QPSK,2450,0,f2450_snr0_QPSK__L05,C2_attn_weight_resid_net,True,33.36109,0.36087,0.21957,1,0.62023,-2.07447,1.87884,0.47262,0.15506,230,83,101,0.5,0.25,-2.41882,0.25,0.125,-2.42075,0.51846,0.05556,-16.42854,0.71069,-1153.77518,119.87045,-0.03419
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:5771:f708_snr20_QPSK__L00__TT012__S406608__stress_low_snr,f708_snr20_QPSK,708,20,f708_snr20_QPSK__L00,C2_attn_weight_resid_net,True,37.72845,0.27391,0.13913,1,0.39805,-4.00065,-2.36073,0.32361,0.09951,230,63,64,0.5,0.25,-0.22811,0.25,0.125,-4.41169,0.46298,0.12939,-12.17604,1.01768,-327.10373,73.90274,0.10842
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:5789:f708_snr20_QPSK__L01__TT016__S430736__stress_low_snr,f708_snr20_QPSK,708,20,f708_snr20_QPSK__L01,C2_attn_weight_resid_net,True,31.5536,0.21739,0.11522,1,0.33602,-4.73637,-3.16383,0.26974,0.084,230,50,53,0.16667,0.08333,-6.37375,0.25,0.125,-1.30582,0.39238,0.0376,-12.43016,0.81375,-393.42133,-12.41486,0.30401
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:5837:f708_snr20_QPSK__L05__TT000__S334224__stress_low_snr,f708_snr20_QPSK,708,20,f708_snr20_QPSK__L05,C2_attn_weight_resid_net,True,35.70359,0.27391,0.15217,1,0.42282,-3.73841,-1.36016,0.33421,0.10571,230,63,70,0,0,-8.81833,0.25,0.125,-8.33886,0.49687,0.00799,-12.49337,0.81237,-383.92,32.08647,1.07709
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:5886:f708_snr10_QPSK__L00__TT005__S364384__stress_cfo_phase_jump,f708_snr10_QPSK,708,10,f708_snr10_QPSK__L00,C2_attn_weight_resid_net,True,39.0068,0.01739,0.0087,1,0.03601,-14.43576,-14.42931,0.04115,0.009,230,4,4,0,0,-26.04875,0,0,-18.86469,0.15128,0.05455,-16.46047,0.55853,-911.80652,-96.55344,-0.13726
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:5902:f708_snr10_QPSK__L01__TT006__S370416__stress_burst_impairment,f708_snr10_QPSK,708,10,f708_snr10_QPSK__L01,C2_attn_weight_resid_net,True,35.75284,0.23478,0.12391,1,0.36961,-4.32257,-2.38847,0.30877,0.0924,230,54,57,0,0,-15.62498,0.25,0.125,-7.33604,0.34776,0.1773,-15.36663,0.70126,-706.78473,-246.82151,0.61554
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:5915:f708_snr10_QPSK__L02__TT004__S358352__stress_low_snr,f708_snr10_QPSK,708,10,f708_snr10_QPSK__L02,C2_attn_weight_resid_net,True,28.78943,0.34348,0.1913,1,0.56139,-2.50735,0.97248,0.43608,0.14035,230,79,88,0,0,-12.75573,0.25,0.125,-2.42349,0.4571,0.13555,-15.12301,0.82216,-433.38789,75.45244,1.05254
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:5945:f708_snr10_QPSK__L04__TT004__S358352__stress_low_snr,f708_snr10_QPSK,708,10,f708_snr10_QPSK__L04,C2_attn_weight_resid_net,True,33.43196,0.31304,0.16739,1,0.47894,-3.1972,-0.71549,0.37541,0.11973,230,72,77,0.33333,0.16667,-4.27559,0,0,-11.21776,0.54684,0.10988,-14.9363,0.92774,-169.86585,68.97448,-0.13603
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:5971:f708_snr10_QPSK__L05__TT018__S442800__stress_burst_impairment,f708_snr10_QPSK,708,10,f708_snr10_QPSK__L05,C2_attn_weight_resid_net,True,34.56716,0.43043,0.23043,1,0.64034,-1.93589,2.36051,0.49202,0.16009,230,99,106,0.5,0.25,-0.90145,0,0,-12.48756,0.79831,0.0333,-14.70118,0.85369,164.14215,-42.4051,-0.99386
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:5991:f708_snr10_QPSK__L07__TT005__S364384__stress_cfo_phase_jump,f708_snr10_QPSK,708,10,f708_snr10_QPSK__L07,C2_attn_weight_resid_net,True,40.07035,0.22609,0.11304,1,0.29145,-5.35429,-3.92462,0.22752,0.07286,230,52,52,0,0,-37.73114,0.25,0.125,-3.93607,0.28132,0.12439,-16.11591,0.64166,-1246.34428,-697.18303,0.61356
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:5992:f708_snr10_QPSK__L07__TT006__S370416__stress_burst_impairment,f708_snr10_QPSK,708,10,f708_snr10_QPSK__L07,C2_attn_weight_resid_net,True,33.64937,0.26522,0.14565,1,0.46122,-3.36089,-0.74519,0.40722,0.11531,230,61,67,0,0,-15.22777,0.5,0.25,2.54558,0.49618,0.08631,-14.92104,0.98234,-389.16783,-156.27794,1
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:5994:f708_snr10_QPSK__L07__TT009__S388512__stress_cfo_phase_jump,f708_snr10_QPSK,708,10,f708_snr10_QPSK__L07,C2_attn_weight_resid_net,True,52.32293,0.12174,0.06087,1,0.20754,-6.82906,-6.17293,0.18702,0.05188,230,28,28,0,0,-50.77388,0.5,0.25,0.28499,0.1507,0.07635,-16.41941,0.35629,-404.93606,640.64077,0.79294
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:6034:f708_snr0_QPSK__L02__TT002__S346288__stress_burst_impairment,f708_snr0_QPSK,708,0,f708_snr0_QPSK__L02,C2_attn_weight_resid_net,True,34.29851,0.68696,0.4413,1,1.16107,0.64857,8.5621,0.81889,0.29027,230,158,203,0.33333,0.25,3.24208,0,0,-7.62508,1.10664,1.44241,-16.67368,1.58234,-1037.36857,-1045.13604,1.125
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:6049:f708_snr0_QPSK__L03__TT002__S346288__stress_burst_impairment,f708_snr0_QPSK,708,0,f708_snr0_QPSK__L03,C2_attn_weight_resid_net,True,61.44123,0.75217,0.52609,1,1.16322,0.65661,20.80817,0.78696,0.2908,230,173,242,0.83333,0.5,9.65739,0.25,0.125,-3.24733,1.08909,1.36195,-17.26191,0.90787,-1396.23606,-1555.4821,-1.06545
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:6095:f708_snr0_QPSK__L06__TT004__S358352__stress_low_snr,f708_snr0_QPSK,708,0,f708_snr0_QPSK__L06,C2_attn_weight_resid_net,True,36.40527,0.60435,0.3413,1,0.87646,-0.57269,5.77503,0.63331,0.21911,230,139,157,0.33333,0.16667,-0.10621,0.25,0.125,0.52018,1.2539,0.36081,-16.9347,1.14879,-12.81867,289.11432,-1.125
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:6107:f708_snr0_QPSK__L07__TT000__S334224__stress_low_snr,f708_snr0_QPSK,708,0,f708_snr0_QPSK__L07,C2_attn_weight_resid_net,True,28.37917,0.52609,0.3,1,0.74153,-1.2987,4.28327,0.54227,0.18538,230,121,138,0.16667,0.08333,-4.07169,0.75,0.375,0.36064,0.49945,0.20026,-17.56882,0.65088,-80.71102,306.22196,-0.19628
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:6140:f433_snr20_QPSK__L01__TT004__S358352__stress_low_snr,f433_snr20_QPSK,433,20,f433_snr20_QPSK__L01,C2_attn_weight_resid_net,True,26.36091,0.2087,0.11304,1,0.33021,-4.81204,-3.09415,0.26838,0.08255,230,48,52,0.16667,0.08333,-9.01121,0.25,0.125,-5.11102,0.38218,0.00897,-12.66824,0.76867,-198.77185,-25.92517,0.01463
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:6167:f433_snr20_QPSK__L03__TT000__S334224__stress_low_snr,f433_snr20_QPSK,433,20,f433_snr20_QPSK__L03,C2_attn_weight_resid_net,True,31.11854,0.24783,0.12609,1,0.36638,-4.36073,-2.65104,0.29855,0.09159,230,57,58,0.16667,0.08333,-5.55717,0.25,0.125,-6.13747,0.43715,0.12397,-12.72873,0.88496,-383.28228,-200.4356,0.36838
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:6170:f433_snr20_QPSK__L03__TT004__S358352__stress_low_snr,f433_snr20_QPSK,433,20,f433_snr20_QPSK__L03,C2_attn_weight_resid_net,True,52.29901,0.15217,0.08261,1,0.24767,-6.0613,-4.99377,0.21645,0.06192,230,35,38,0,0,-13.62886,0,0,-8.44053,0.36017,0.01201,-13.12234,0.47122,-143.92954,38.91714,0.56305
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:6203:f433_snr20_QPSK__L05__TT008__S382480__stress_low_snr,f433_snr20_QPSK,433,20,f433_snr20_QPSK__L05,C2_attn_weight_resid_net,True,32.09144,0.38261,0.21739,1,0.59713,-2.23933,1.62478,0.44835,0.14928,230,88,100,0.16667,0.16667,-0.74751,0.25,0.125,-4.9461,0.57882,0.08373,-12.31905,1.11823,-364.74998,-201.9033,-0.93609
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:6223:f433_snr20_QPSK__L06__TT014__S418672__stress_burst_impairment,f433_snr20_QPSK,433,20,f433_snr20_QPSK__L06,C2_attn_weight_resid_net,True,36.09798,0.24783,0.13696,1,0.40208,-3.95693,-1.93436,0.32826,0.10052,230,57,63,0,0,-10.25421,0.25,0.125,-5.35317,0.46298,0.00923,-12.44754,1.39042,-216.60877,285.36081,0.97311
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:6249:f433_snr10_QPSK__L00__TT009__S388512__stress_cfo_phase_jump,f433_snr10_QPSK,433,10,f433_snr10_QPSK__L00,C2_attn_weight_resid_net,True,30.63955,0,0,0,0.00373,-24.28367,-24.41212,0.00715,0.00093234,230,0,0,0,0,-43.92062,0,0,-45.62129,0.10042,0.06886,-16.66428,0.36088,364.67156,85.06308,-0.19805
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:6289:f433_snr10_QPSK__L03__TT002__S346288__stress_burst_impairment,f433_snr10_QPSK,433,10,f433_snr10_QPSK__L03,C2_attn_weight_resid_net,True,66.59815,0.2087,0.10652,1,0.44596,-3.50707,-2.47287,0.38309,0.11149,230,48,49,0.33333,0.16667,-3.09351,0.5,0.375,0.88227,0.36021,0.16856,-15.07293,0.42849,-276.03647,-195.39744,1.125
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:6316:f433_snr10_QPSK__L04__TT018__S442800__stress_burst_impairment,f433_snr10_QPSK,433,10,f433_snr10_QPSK__L04,C2_attn_weight_resid_net,True,29.30087,0.18696,0.10217,1,0.34466,-4.62608,-2.93775,0.30119,0.08617,230,43,47,0.16667,0.08333,-4.81931,0,0,-12.88586,0.30844,0.03601,-15.36808,0.6068,76.53728,-100.77791,-0.28379
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:6320:f433_snr10_QPSK__L05__TT004__S358352__stress_low_snr,f433_snr10_QPSK,433,10,f433_snr10_QPSK__L05,C2_attn_weight_resid_net,True,66.47915,0.23913,0.12826,1,0.36047,-4.43135,-2.5388,0.29344,0.09012,230,55,59,0,0,-6.86234,0,0,-19.48084,0.38482,0.03526,-15.21629,0.52153,-317.06246,-102.71736,-0.04165
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:6414:f433_snr0_QPSK__L03__TT009__S388512__stress_cfo_phase_jump,f433_snr0_QPSK,433,0,f433_snr0_QPSK__L03,C2_attn_weight_resid_net,True,32.44498,0.36087,0.18261,1,0.51993,-2.84054,-1.13083,0.41998,0.12998,230,83,84,0,0,-18.48175,0,0,-8.89271,0.7092,0.30318,-16.26759,1.89804,-303.72038,-613.25491,-0.02421
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:6416:f433_snr0_QPSK__L03__TT012__S406608__stress_low_snr,f433_snr0_QPSK,433,0,f433_snr0_QPSK__L03,C2_attn_weight_resid_net,True,62.98973,0.33913,0.19565,1,0.58124,-2.35646,0.91781,0.45047,0.14531,230,78,90,0.33333,0.16667,-5.23492,0,0,-17.12226,0.4026,0.17631,-16.38347,0.54935,42.36083,215.67077,0.69729
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:6431:f433_snr0_QPSK__L04__TT012__S406608__stress_low_snr,f433_snr0_QPSK,433,0,f433_snr0_QPSK__L04,C2_attn_weight_resid_net,True,39.21842,0.36087,0.19783,1,0.57934,-2.37066,1.33346,0.45018,0.14484,230,83,91,0.16667,0.08333,-1.40102,0,0,-13.46253,0.51551,0.01265,-15.81133,0.85102,-240.24368,-86.93374,0.91493
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:6451:f433_snr0_QPSK__L05__TT018__S442800__stress_burst_impairment,f433_snr0_QPSK,433,0,f433_snr0_QPSK__L05,C2_attn_weight_resid_net,True,31.87467,0.36087,0.2,1,0.60591,-2.17589,1.79622,0.48404,0.15148,230,83,92,0.33333,0.16667,-2.14563,0.25,0.125,-6.22999,0.54492,0.09372,-15.96488,0.6712,-474.93472,99.53809,1.125
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/per_packet_test_stress.csv:6472:f433_snr0_QPSK__L07__TT006__S370416__stress_burst_impairment,f433_snr0_QPSK,433,0,f433_snr0_QPSK__L07,C2_attn_weight_resid_net,True,27.55394,0.36087,0.19348,1,0.60363,-2.19228,0.6195,0.47566,0.15091,230,83,89,0.33333,0.16667,-3.05202,0.25,0.125,-3.39333,0.908,0.8547,-15.62231,0.55484,-297.74003,-141.6628,0.77397
origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/run_pipeline.py:729:    # Create stress-perturbed train (subset 720), dev (432), and test (1080 = 360 per stress mode across all 9 captures)
REMOTE_REPO = Path("/workspace/EM-FloatFormer")
REMOTE_OUT = Path("/vol/stage3d")


def _setup_remote_env():
    os.chdir(REMOTE_REPO)
    if str(REMOTE_REPO) not in sys.path:
        sys.path.insert(0, str(REMOTE_REPO))


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=1800,
    volumes={"/root/.cache/huggingface": hf_vol, "/vol/stage3d": res_vol},
)
def remote_smoke_qwen() -> dict:
    _setup_remote_env()
    from research.stage3d_qwen_signed_phase.train_eval_stage3d import run_smoke_audit

    out_dir = REMOTE_OUT / "results"
    report = run_smoke_audit(REMOTE_REPO, out_dir)
    hf_vol.commit()
    res_vol.commit()
    return report


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=7200,
    volumes={"/root/.cache/huggingface": hf_vol, "/vol/stage3d": res_vol},
)
def remote_train_one(
    arm: str,
    seed: int,
    epochs: int = 10,
    batch_size: int = 32,
    micro_batch_size: int = 8,
    lr: float = 1.5e-3,
    max_steps_per_epoch: int = 200,
    train_blocks: int = 8,
    dev_blocks: int = 2,
    train_scorer: str = "kv_cached",
    eval_scorer: str = "kv_cached",
```

### Event #31: `item.completed` — `file_change` (`item_28`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M4\\verify_m4.py",
    "kind": "add"
  }
]
```

### Event #32: `item.completed` — `command_execution` (`item_29`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command 'git show origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/evaluator.py | Select-Object -First 75; git show origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/run_pipeline.py | Select-Object -Skip 725 -First 47; Get-ChildItem -Name research/stage4_math_and_design/M1,research/stage4_math_and_design/M2,research/stage4_math_and_design/M3'
```

**Aggregated Output:**

```text
"""Unified evaluation, calibration, and cluster-paired statistical inference module.

Implements strict segmented scoring across sync header, in-packet pilots, withheld holdout
pilots, and payload symbols/bits, plus engineering cost and coverage accounting.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


# Gray-mapped QPSK constellation verified empirically against CTTC f2450_snr20_QPSK:
# (b0, b1) -> ((1 - 2*b0) + 1j * (1 - 2*b1)) / sqrt(2)
# Index order: 0:(0,0)->(+1+1j)/s2, 1:(0,1)->(+1-1j)/s2, 2:(1,0)->(-1+1j)/s2, 3:(1,1)->(-1-1j)/s2
INV_SQRT2 = 1.0 / math.sqrt(2.0)
QPSK_CONSTELLATION = np.array(
    [
        (1.0 + 1.0j) * INV_SQRT2,
        (1.0 - 1.0j) * INV_SQRT2,
        (-1.0 + 1.0j) * INV_SQRT2,
        (-1.0 - 1.0j) * INV_SQRT2,
    ],
    dtype=np.complex128,
)
QPSK_BITS_TABLE = np.array(
    [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ],
    dtype=np.uint8,
)


def bits_to_qpsk(bits: np.ndarray) -> np.ndarray:
    """Map unpacked bits (length 2*N, values in {0, 1}) to unit-energy Gray QPSK symbols."""
    b = np.asarray(bits, dtype=np.int64)
    if b.ndim != 1 or (len(b) % 2) != 0:
        raise ValueError(f"Expected 1-D even-length bit array, got shape {b.shape}")
    b0 = b[0::2]
    b1 = b[1::2]
    return ((1.0 - 2.0 * b0) + 1j * (1.0 - 2.0 * b1)) * INV_SQRT2


def qpsk_to_bits(syms: np.ndarray) -> np.ndarray:
    """Hard-demap complex QPSK symbols to unpacked Gray bits [b0_0, b1_0, b0_1, b1_1, ...]."""
    s = np.asarray(syms, dtype=np.complex128)
    out = np.empty(2 * len(s), dtype=np.uint8)
    out[0::2] = (s.real < 0.0).astype(np.uint8)
    out[1::2] = (s.imag < 0.0).astype(np.uint8)
    return out


def qpsk_hard_symbols(syms: np.ndarray) -> np.ndarray:
    """Hard-decide complex equalized symbols onto the unit-energy QPSK constellation."""
    s = np.asarray(syms, dtype=np.complex128)
    re = np.where(s.real >= 0.0, 1.0, -1.0)
    im = np.where(s.imag >= 0.0, 1.0, -1.0)
    return (re + 1j * im) * INV_SQRT2


def qpsk_bit_llrs_gaussian(
    y: np.ndarray,
    h: np.ndarray,
    noise_var: np.ndarray | float,
) -> np.ndarray:
    """Compute exact per-bit LLR = log P(b=1|y) / P(b=0|y) under y = h*x + n, n ~ CN(0, noise_var).

    Returns array of shape (2 * len(y),) interleaved as [b0_0, b1_0, b0_1, b1_1, ...].
    """
    stress_modes = ("stress_low_snr", "stress_cfo_phase_jump", "stress_burst_impairment")
    rng_st = np.random.default_rng(2026112199)

    # Create stress-perturbed train (subset 720), dev (432), and test (1080 = 360 per stress mode across all 9 captures)
    st_train_pkts, st_train_pools = [], []
    for i, (p, pl) in enumerate(zip(train_pkts[::2], train_pools[::2])):
        md = stress_modes[i % 3]
        st_train_pkts.append(apply_controlled_stress(p, rng_st, mode=md))
        st_train_pools.append(pl)

    st_dev_pkts, st_dev_pools = [], []
    for i, (p, pl) in enumerate(zip(dev_pkts[::2], dev_pools[::2])):
        md = stress_modes[i % 3]
        st_dev_pkts.append(apply_controlled_stress(p, rng_st, mode=md))
        st_dev_pools.append(pl)

    st_test_pkts, st_test_pools = [], []
    # Every test packet in a balanced 1,080-packet suite (40/cap x 9 caps x 3 modes -> actually 120/cap = 1080)
    for i, (p, pl) in enumerate(zip(test_pkts, test_pools)):
        if (i % 4) == 3:  # 1,080 packets (120 per capture, 360 per stress mode, balanced across all 72 clusters)
            continue
        md = stress_modes[len(st_test_pkts) % 3]
        st_test_pkts.append(apply_controlled_stress(p, rng_st, mode=md))
        st_test_pools.append(pl)

    st_outputs: dict[str, list[ReceiverOutput]] = {}
    for cmode in ("ls", "mmse", "mmse_slope"):
        st_outputs[f"classical_{cmode}"] = [run_classical_receiver(p, best_cfg, mode=cmode) for p in st_test_pkts]
    for p, pl in zip(st_test_pkts, st_test_pools):
        ao, _ = run_experiment_a_arms(p, pl, best_cfg, local_k=4, content_temp=tau_a)
        for k in ("A1_local_window_k4", "A2_global_uniform", "A3_global_snr_weight", "A4_global_content_attn"):
            st_outputs.setdefault(k, []).append(ao[k])

    st_banks_tr = [build_common_candidate_bank(p, pl, best_cfg, content_temp=tau_a) for p, pl in zip(st_train_pkts, st_train_pools)]
    st_banks_dv = [build_common_candidate_bank(p, pl, best_cfg, content_temp=tau_a) for p, pl in zip(st_dev_pkts, st_dev_pools)]
    st_banks_te = [build_common_candidate_bank(p, pl, best_cfg, content_temp=tau_a) for p, pl in zip(st_test_pkts, st_test_pools)]

    for bk in st_banks_te:
        bo = run_experiment_b_arms(bk, post_temp=tau_b, block_temp=blk_t)
        for k, v in bo.items():
            st_outputs.setdefault(k, []).append(v)

    res_c_stress = train_and_infer_experiment_c(
        st_train_pkts, st_banks_tr, st_dev_pkts, st_banks_dv, st_test_pkts, st_banks_te,
        init_temp=tau_b, block_temp=blk_t, seeds=(20261121, 20261122, 20261123), epochs=45, batch_size=64, lr=4e-3,
    )
    c_stress_multi_seed: dict[str, Any] = {}
.uv-cache
.uv-cache
verify_m2.py
.uv-cache
```

### Event #33: `item.completed` — `file_change` (`item_30`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M4\\verify_m4.py",
    "kind": "update"
  }
]
```

### Event #34: `item.completed` — `file_change` (`item_31`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M4\\verify_m4.py",
    "kind": "update"
  }
]
```

### Event #35: `item.completed` — `file_change` (`item_32`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M4\\verify_m4.py",
    "kind": "update"
  }
]
```

### Event #36: `item.completed` — `command_execution` (`item_33`)

- Status: `failed`, Exit Code: `1`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$env:UV_CACHE_DIR=(Resolve-Path '"'research/stage4_math_and_design/M4/.uv-m4').Path; "'$env:UV_OFFLINE='"'1'; "'$env:UV_NO_PROJECT='"'1'; "'$env:OPENBLAS_NUM_THREADS='"'1'; uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B research/stage4_math_and_design/M4/verify_m4.py"
```

**Aggregated Output:**

```text
error: Failed to install: numpy-2.3.5-cp312-cp312-win_amd64.whl (numpy==2.3.5)
  Caused by: 拒绝访问。 (os error 5) at path "C:\\WINDOWS\\TEMP\\.tmpuVNNB1"
```

### Event #37: `item.completed` — `command_execution` (`item_34`)

- Status: `failed`, Exit Code: `1`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$env:UV_CACHE_DIR=(Resolve-Path '"'research/stage4_math_and_design/M4/.uv-m4').Path; "'$env:TEMP=$env:UV_CACHE_DIR; $env:TMP=$env:UV_CACHE_DIR; $env:UV_OFFLINE='"'1'; "'$env:UV_NO_PROJECT='"'1'; "'$env:UV_LINK_MODE='"'copy'; "'$env:OPENBLAS_NUM_THREADS='"'1'; uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B research/stage4_math_and_design/M4/verify_m4.py"
```

**Aggregated Output:**

```text
Installed 2 packages in 8.02s
Verified frozen s17 identity and pinned Python/NumPy/SciPy
Traceback (most recent call last):
  File "C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M4\verify_m4.py", line 566, in <module>
    main()
  File "C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M4\verify_m4.py", line 544, in main
    manifests = verify_manifests(args.check_http)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M4\verify_m4.py", line 453, in verify_manifests
    assert c['zenodo_record'] == record and c['files'] == c_code['MANIFEST_FILES']
           ~^^^^^^^^^^^^^^^^^
KeyError: 'zenodo_record'
```

### Event #38: `item.completed` — `command_execution` (`item_35`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "git show origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/data_manifest.json | Select-Object -First 45; Get-Item 'research/stage4_math_and_design/M4/.uv-m4' | Format-List FullName,Attributes"
```

**Aggregated Output:**

```text
{
  "dataset_title": "Datasets of Indoor Wireless Channel Measurements for Machine Learning Applications",
  "authors": [
    {
      "name": "Adriano Pastore",
      "affiliation": "Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)"
    },
    {
      "name": "Armin Ghani",
      "affiliation": "Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)"
    }
  ],
  "publication_date": "2021-06-02",
  "project_grant": "ARISTIDES (RTI2018-099722-B-I00, Spanish Ministry of Economy and Competitiveness)",
  "license": "CC-BY-4.0",
  "zenodo_record_id": "4895133",
  "doi": "10.5281/zenodo.4895133",
  "url": "https://zenodo.org/records/4895133",
  "api_url": "https://zenodo.org/api/records/4895133",
  "files": [
    {
      "filename": "README.pdf",
      "role": "documentation",
      "size_bytes": 1549089,
      "md5_official_verified": "620c116c2a38a6dfbb6316752833850d",
      "sha256": "c8e7ae57e0896dfe2f1cc910d084e9b77e3cc5143b721cf9b24753ab8b496fb4"
    },
    {
      "filename": "sources.tar.gz",
      "role": "tx_source_bits",
      "size_bytes": 157051651,
      "md5_official_verified": "48766e3f5ddaa968ccf128eb5891a450",
      "sha256": "f9022be37d3eeadcff481f46b25ca9a067c0e19988bc4bdcb77cca9957ee7b4f",
      "extracted": {
        "source_QPSK.dat": {
          "size_bytes": 1000000,
          "sha256": "6dc9635b72b34f2d792f7ec6522a0450181397fb616262b629ee593e35f3f1cb"
        }
      }
    },
    {
      "filename": "f2450_snr20_QPSK.tar.gz",
      "capture_id": "f2450_snr20_QPSK",
      "freq_mhz": 2450,
      "nom_snr_db": 20,


FullName   : C:\Users\������\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M4\.uv-m
             4
Attributes : Directory



```

### Event #39: `item.completed` — `command_execution` (`item_36`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command 'git show origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/data_manifest.json | Select-Object -Skip 42 -First 32; git show origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/data_manifest.json | Select-Object -Last 12'
```

**Aggregated Output:**

```text
      "capture_id": "f2450_snr20_QPSK",
      "freq_mhz": 2450,
      "nom_snr_db": 20,
      "size_bytes": 211697276,
      "md5_official_verified": "1df69db08519d130faff0857347930de",
      "sha256": "161d618ef2da2412a3c4d7236deb7c52f0b81f31af704b9d852d90f849c6cbb3",
      "extracted": {
        "config.txt": {"size_bytes": 287, "sha256": "7e9e6ff2af0f10f2cf0a47679394a6443a1fb8051c6e0c61acacd29c55a5389e"},
        "file_1.dat": {"size_bytes": 176015264, "sha256": "c6596f98fd9705c475b8b07340facb6ad7dbc4906e6f196ac2f207a275544ab1"},
        "file_2.dat": {"size_bytes": 176015264, "sha256": "20c521344958acda0cd30dcd0449a41ab16d472e6942bee0d5e539c788c7a4db"}
      }
    },
    {
      "filename": "f2450_snr10_QPSK.tar.gz",
      "capture_id": "f2450_snr10_QPSK",
      "freq_mhz": 2450,
      "nom_snr_db": 10,
      "size_bytes": 197835355,
      "md5_official_verified": "545e2f96b666200da371bb51726bb336",
      "sha256": "e6b7462ec66d485b1b10936e3f86601c91e900dddad9b8b9e19b789c3e183d25",
      "extracted": {
        "config.txt": {"size_bytes": 287, "sha256": "b9dde9083a6b228929913b577f2bfc2e2e9e50a61638a0dbe68bc7fccf6cd221"},
        "file_1.dat": {"size_bytes": 176015264, "sha256": "3d56b0f14b97755ecdfae2011f011165a6837c2caa363a9cbfaedc0dfcd53d0c"},
        "file_2.dat": {"size_bytes": 176015264, "sha256": "b91aba4447771e7dfd149515067ee64e1303ae44c6669d65504f17899070f331"}
      }
    },
    {
      "filename": "f2450_snr0_QPSK.tar.gz",
      "capture_id": "f2450_snr0_QPSK",
      "freq_mhz": 2450,
      "nom_snr_db": 0,
      "size_bytes": 191552426,
      "nom_snr_db": 0,
      "size_bytes": 193721020,
      "md5_official_verified": "9cc2a791ef1b07977b5c43760a06bda8",
      "sha256": "c7b7bcba97151ad4dadebe3c21fb3503d44126c6912e56e024a19d303b17ef2f",
      "extracted": {
        "config.txt": {"size_bytes": 285, "sha256": "8f9b54ba0153ae3f9e53a6ba6e231708934c7a1b42957120ee122ebe5ec9aeca"},
        "file_1.dat": {"size_bytes": 176015264, "sha256": "84f47c34c7daac92412742af7133ac984b2e713408b8b6e57665708c07a64e43"},
        "file_2.dat": {"size_bytes": 176015264, "sha256": "98d19a36fbc990d9fd7214eb7855482703f76650b5fca7ef1d513730e9541d54"}
      }
    }
  ]
}
```

### Event #40: `item.completed` — `file_change` (`item_37`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M4\\verify_m4.py",
    "kind": "update"
  }
]
```

### Event #41: `item.completed` — `command_execution` (`item_38`)

- Status: `failed`, Exit Code: `1`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$env:UV_CACHE_DIR=(Resolve-Path '"'research/stage4_math_and_design/M4/.uv-m4').Path; "'$env:TEMP=$env:UV_CACHE_DIR; $env:TMP=$env:UV_CACHE_DIR; $env:UV_OFFLINE='"'1'; "'$env:UV_NO_PROJECT='"'1'; "'$env:UV_LINK_MODE='"'copy'; "'$env:OPENBLAS_NUM_THREADS='"'1'; uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B research/stage4_math_and_design/M4/verify_m4.py"
```

**Aggregated Output:**

```text
Verified frozen s17 identity and pinned Python/NumPy/SciPy
Verified static splits, historical manifests and Modal arithmetic
Traceback (most recent call last):
  File "C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M4\verify_m4.py", line 574, in <module>
    main()
  File "C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M4\verify_m4.py", line 556, in main
    distributions = verify_distributions(rng)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M4\verify_m4.py", line 137, in verify_distributions
    np.testing.assert_allclose(radial_kl, exact, atol=1e-8)
  File "C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M4\.uv-m4\archive-v0\7UHkdbs1IKEY1XnX\Lib\site-packages\numpy\testing\_private\utils.py", line 1711, in assert_allclose
    assert_array_compare(compare, actual, desired, err_msg=str(err_msg),
  File "C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M4\.uv-m4\archive-v0\7UHkdbs1IKEY1XnX\Lib\site-packages\numpy\testing\_private\utils.py", line 919, in assert_array_compare
    raise AssertionError(msg)
AssertionError: 
Not equal to tolerance rtol=1e-07, atol=1e-08

Mismatched elements: 1 / 1 (100%)
Max absolute difference among violations: 1.
Max relative difference among violations: 2.
 ACTUAL: array(0.5)
 DESIRED: array(-0.5)
```

### Event #42: `item.completed` — `file_change` (`item_39`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M4\\verify_m4.py",
    "kind": "update"
  }
]
```

### Event #43: `item.completed` — `command_execution` (`item_40`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$env:UV_CACHE_DIR=(Resolve-Path '"'research/stage4_math_and_design/M4/.uv-m4').Path; "'$env:TEMP=$env:UV_CACHE_DIR; $env:TMP=$env:UV_CACHE_DIR; $env:UV_OFFLINE='"'1'; "'$env:UV_NO_PROJECT='"'1'; "'$env:UV_LINK_MODE='"'copy'; "'$env:OPENBLAS_NUM_THREADS='"'1'; uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B research/stage4_math_and_design/M4/verify_m4.py"
```

**Aggregated Output:**

```text
Verified frozen s17 identity and pinned Python/NumPy/SciPy
Verified static splits, historical manifests and Modal arithmetic
PASS 168 mapping cells, 32 nuisance cells, results_m4.json
```

### Event #44: `item.completed` — `command_execution` (`item_41`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n 'Pf|F0|group|rotate|expected' research/stage3d_qwen_signed_phase/analysis_stage3d.py | Select-Object -First 50; rg --files research/stage4_math_and_design/M1 research/stage4_math_and_design/M2 research/stage4_math_and_design/M3 -g '*.md'"
```

**Aggregated Output:**

```text
139:            v1, v2 = batch.rotated_view(s1, s2)
151:        v1_10, v2_10 = batch.rotated_view(1, 0)
206:            v1, v2 = batch.rotated_view(s1, s2)
425:                v1, v2 = batch.rotated_view(s1, s2)
research/stage4_math_and_design/M2\THEOREM_M2.md
```

### Event #45: `item.completed` — `command_execution` (`item_42`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$r=Get-Content -Raw research/stage4_math_and_design/M4/results_m4.json | ConvertFrom-Json; $r.dataset_manifests | Select-Object total_download_bytes; $r.dataset_manifests.cttc | Select-Object capture_archives_bytes,all_selected_download_bytes,selected_extracted_bytes; $r.index_audit.cttc | Select-Object minimum_acquisition_gap_samples_after_rcv_support_expansion; $r.modal_budget.rows | Format-Table; $r.distributions.correlated_complex_gaussian_kl; $r.distributions.student_t_radial | Select-Object m,nu,kl_to_nominal,kl_to_moment_matched_gaussian; foreach ($c in $r.frozen_encoder_sweep.cells) { if (($c.dc -eq 0 -and $c.spur -eq 0 -and $c.delta -eq 0) -or ($c.dc -eq 0.30 -and $c.spur -eq 0.26 -and $c.delta -eq 0.30 -and $c.physical_snr_db -in -2,2,20)) { foreach ($m in $c.maps.PSObject.Properties) { [PSCustomObject]@{SNR=$c.physical_snr_db;delta=$c.delta; dc=$c.dc; spur=$c.spur; map=$m.Name; SER=[math]::Round($m.Value.ser,5); NLL=[math]::Round($m.Value.frozen_table_nll_nats,5); MI=[math]::Round($m.Value.code_uniform_mi_nats,5)} } } }; $r.frozen_encoder_sweep.ray_check; $r.nuisance_curves.conditional_curves | Where-Object {($_.kind -eq '"'coherent_spur' -and "'$_.value -eq 0.26) -or ($_.kind -eq '"'cfo_rad_per_symbol' -and "'$_.value -eq 0.02) -or ($_.kind -eq '"'student_t_nu' -and "'$_.value -in 1,3) } | Select-Object kind,value,lag_symbols,conditional_joint_kl_nats,scalar_oracle_uniform_mi_nats_mc,nominal_posterior_regret_nats_mc'
```

**Aggregated Output:**

```text

total_download_bytes
--------------------
          2789188515
                    
                    



phase                     resource  seconds_cap physical_cores_cap gib_cap usd_cap   
-----                     --------  ----------- ------------------ ------- -------   
connector_train           A100-80GB        3840                  2       8 2.83376640
real_raw                  A100-80GB        2477                  2       8 1.82792692
real_stress_q2            A100-80GB         248                  2       8 0.18301408
synthetic_blind           A100-80GB        1229                  2       8 0.90695284
dev_calibration           A100-80GB         480                  2       8 0.35422080
gpu_smoke_boot_and_commit A100-80GB        1100                  2       8 0.81175600
gpu_retry_reserve         A100-80GB         900                  2       8 0.66416400
ingest_and_verify         CPU              1800                  1       8 0.05554800
frontend_and_physical     CPU              7200                  4      16 0.63302400
audit_and_aggregation     CPU              1800                  2       8 0.07912800



       monte_carlo                mc_se             exact
       -----------                -----             -----
1.1984601709141602 0.004143577620340218 1.196255370544037
                                                         
                                                         
                                                         
                                                         
                                                         
                                                         
                                                         
                                                         
                                                         
                                                         
                                                         
                                                         
                                                         
                                                         
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:2 �ַ�: 809
+ ... operties) { [PSCustomObject]@{SNR=$c.physical_snr_db;delta=$c.delta;  ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:2 �ַ�: 809
+ ... operties) { [PSCustomObject]@{SNR=$c.physical_snr_db;delta=$c.delta;  ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:2 �ַ�: 809
+ ... operties) { [PSCustomObject]@{SNR=$c.physical_snr_db;delta=$c.delta;  ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:2 �ַ�: 809
+ ... operties) { [PSCustomObject]@{SNR=$c.physical_snr_db;delta=$c.delta;  ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:2 �ַ�: 809
+ ... operties) { [PSCustomObject]@{SNR=$c.physical_snr_db;delta=$c.delta;  ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:2 �ַ�: 809
+ ... operties) { [PSCustomObject]@{SNR=$c.physical_snr_db;delta=$c.delta;  ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:2 �ַ�: 809
+ ... operties) { [PSCustomObject]@{SNR=$c.physical_snr_db;delta=$c.delta;  ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:2 �ַ�: 809
+ ... operties) { [PSCustomObject]@{SNR=$c.physical_snr_db;delta=$c.delta;  ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:2 �ַ�: 809
+ ... operties) { [PSCustomObject]@{SNR=$c.physical_snr_db;delta=$c.delta;  ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:2 �ַ�: 809
+ ... operties) { [PSCustomObject]@{SNR=$c.physical_snr_db;delta=$c.delta;  ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:2 �ַ�: 809
+ ... operties) { [PSCustomObject]@{SNR=$c.physical_snr_db;delta=$c.delta;  ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:2 �ַ�: 809
+ ... operties) { [PSCustomObject]@{SNR=$c.physical_snr_db;delta=$c.delta;  ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:2 �ַ�: 809
+ ... operties) { [PSCustomObject]@{SNR=$c.physical_snr_db;delta=$c.delta;  ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:2 �ַ�: 809
+ ... operties) { [PSCustomObject]@{SNR=$c.physical_snr_db;delta=$c.delta;  ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:2 �ַ�: 809
+ ... operties) { [PSCustomObject]@{SNR=$c.physical_snr_db;delta=$c.delta;  ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:2 �ַ�: 809
+ ... operties) { [PSCustomObject]@{SNR=$c.physical_snr_db;delta=$c.delta;  ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:2 �ַ�: 809
+ ... operties) { [PSCustomObject]@{SNR=$c.physical_snr_db;delta=$c.delta;  ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:2 �ַ�: 809
+ ... operties) { [PSCustomObject]@{SNR=$c.physical_snr_db;delta=$c.delta;  ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:2 �ַ�: 809
+ ... operties) { [PSCustomObject]@{SNR=$c.physical_snr_db;delta=$c.delta;  ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:2 �ַ�: 809
+ ... operties) { [PSCustomObject]@{SNR=$c.physical_snr_db;delta=$c.delta;  ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:2 �ַ�: 809
+ ... operties) { [PSCustomObject]@{SNR=$c.physical_snr_db;delta=$c.delta;  ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:2 �ַ�: 809
+ ... operties) { [PSCustomObject]@{SNR=$c.physical_snr_db;delta=$c.delta;  ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:2 �ַ�: 809
+ ... operties) { [PSCustomObject]@{SNR=$c.physical_snr_db;delta=$c.delta;  ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:2 �ַ�: 809
+ ... operties) { [PSCustomObject]@{SNR=$c.physical_snr_db;delta=$c.delta;  ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:2 �ַ�: 809
+ ... operties) { [PSCustomObject]@{SNR=$c.physical_snr_db;delta=$c.delta;  ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:2 �ַ�: 809
+ ... operties) { [PSCustomObject]@{SNR=$c.physical_snr_db;delta=$c.delta;  ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:2 �ַ�: 809
+ ... operties) { [PSCustomObject]@{SNR=$c.physical_snr_db;delta=$c.delta;  ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:2 �ַ�: 809
+ ... operties) { [PSCustomObject]@{SNR=$c.physical_snr_db;delta=$c.delta;  ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:2 �ַ�: 809
+ ... operties) { [PSCustomObject]@{SNR=$c.physical_snr_db;delta=$c.delta;  ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:2 �ַ�: 809
+ ... operties) { [PSCustomObject]@{SNR=$c.physical_snr_db;delta=$c.delta;  ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:2 �ַ�: 809
+ ... operties) { [PSCustomObject]@{SNR=$c.physical_snr_db;delta=$c.delta;  ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:2 �ַ�: 809
+ ... operties) { [PSCustomObject]@{SNR=$c.physical_snr_db;delta=$c.delta;  ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:2 �ַ�: 809
+ ... operties) { [PSCustomObject]@{SNR=$c.physical_snr_db;delta=$c.delta;  ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:2 �ַ�: 809
+ ... operties) { [PSCustomObject]@{SNR=$c.physical_snr_db;delta=$c.delta;  ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:2 �ַ�: 809
+ ... operties) { [PSCustomObject]@{SNR=$c.physical_snr_db;delta=$c.delta;  ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:2 �ַ�: 809
+ ... operties) { [PSCustomObject]@{SNR=$c.physical_snr_db;delta=$c.delta;  ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:2 �ַ�: 809
+ ... operties) { [PSCustomObject]@{SNR=$c.physical_snr_db;delta=$c.delta;  ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:2 �ַ�: 809
+ ... operties) { [PSCustomObject]@{SNR=$c.physical_snr_db;delta=$c.delta;  ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:2 �ַ�: 809
+ ... operties) { [PSCustomObject]@{SNR=$c.physical_snr_db;delta=$c.delta;  ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:2 �ַ�: 809
+ ... operties) { [PSCustomObject]@{SNR=$c.physical_snr_db;delta=$c.delta;  ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
                                                         
                                                         
                                                         
                                                         
                                                         
                                                         
                                                         


```

### Event #46: `item.completed` — `command_execution` (`item_43`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command 'Get-Content -Encoding utf8 -TotalCount 65 research/stage4_math_and_design/M2/THEOREM_M2.md; $r=Get-Content -Raw research/stage4_math_and_design/M4/results_m4.json | ConvertFrom-Json; $r.dataset_manifests.cttc | Select-Object capture_archives_bytes,all_selected_download_bytes,selected_extracted_bytes | Format-List; $r.index_audit.cttc.minimum_acquisition_gap_samples_after_rcv_support_expansion; $r.distributions.student_t_radial | Format-Table m,nu,kl_to_nominal,kl_to_moment_matched_gaussian; $r.frozen_encoder_sweep.ray_check | ConvertTo-Json -Compress; foreach ($c in $r.frozen_encoder_sweep.cells) { if (($c.dc -eq 0 -and $c.spur -eq 0 -and $c.delta -eq 0 -and $c.physical_snr_db -in -2,2,20) -or ($c.dc -eq 0.30 -and $c.spur -eq 0.26 -and $c.delta -eq 0.30 -and $c.physical_snr_db -in -2,2,20)) { foreach ($m in $c.maps.PSObject.Properties) { '"'{0,3} {1} {2,-23} {3:F5} {4:F5} {5:F5}' -f "'$c.physical_snr_db,$c.delta,$m.Name,$m.Value.ser,$m.Value.frozen_table_nll_nats,$m.Value.code_uniform_mi_nats } } }; $r.nuisance_curves.conditional_curves | Where-Object {($_.kind -eq '"'coherent_spur' -and "'$_.value -eq 0.26) -or ($_.kind -eq '"'cfo_rad_per_symbol' -and "'$_.value -eq 0.02) -or ($_.kind -eq '"'student_t_nu' -and "'$_.value -in 1,3) } | Format-Table kind,value,lag_symbols,conditional_joint_kl_nats,scalar_oracle_uniform_mi_nats_mc,nominal_posterior_regret_nats_mc'
```

**Aggregated Output:**

```text
# Track M2 | Stage 3D score forensics and the Stage 4 gate

Round 1. Losses and information are in nats. Positive paired differences favor phase, so the accuracy contrast is phase minus zero and the NLL contrast is zero minus phase. Each saved mother carries its signs and rotations together. Each of the three connector seeds has equal weight. All calibration fitted on blind labels is retrospective.

## 1. Which Bayes risk is being estimated?

For a population law \(P\), write \(p(c)=P(Y=\cdot\mid C=c)\) and \(L(q)=\mathbb E[-\log q_Y(C)]\). With finite support and positive model probabilities,

\[
L(q)=H(Y\mid C)+\mathbb E_C D_{\rm KL}\!\left(p(C)\Vert q(C)\right).
\]

This follows by adding and subtracting \(-\log p_Y(C)\) inside the conditional expectation. It is the Bregman decomposition for negative Shannon entropy. If there is a finer observation \(X\) and \(C\) is a function of \(X\), then

\[
L(q)=H(Y\mid X)+I(Y;X\mid C)+\mathbb E_C D_{\rm KL}(p(C)\Vert q(C)).
\]

The population entropy, the information lost by coarsening, and the conditional forecast error depend on the named observation. In Q3, \(C=(S,Z)\) is the physical/told CFO sign and the shared S17 code pair at a fixed condition. Both arms are functions of \(C\) and the fitted seed \(J\). Under the empirical law \(J\) is uniform and independent of \((C,Y)\). Therefore \(H(Y\mid C,J)=H(Y\mid C)\). The same Bayes term cancels exactly in the arm contrast. In particular the Q3 input-resolution difference is zero. Improving a forecast can still use existing information more effectively.

The verifier uses three distinct reference laws.

- **Empirical signed-code posterior.** Entropy of blind counts pooled over the prescribed three rotations, separately by \((S,Z)\). It gives an exact replay identity and an in-sample Bayes risk for that empirical law. Estimation and scoring reuse the labels. Sparse codes and especially H make the plug-in conditional MI optimistic as population evidence.
- **S17 direct readout.** A physical predictive baseline using encoder representatives and known condition. Its realized loss is scored on the same mothers. The code does not prove that integrating the population distribution over a quantizer cell equals scoring the representative.
- **Known-parameter full IQ.** A finer physical reference. Its mean soft entropy and realized NLL differ in finite samples. The identity \(L_{\rm realized}(q)=\overline{H(r)}+\overline{D_{\rm KL}(r\Vert q)}+\overline{\sum_y(r_y-1_{Y=y})\log q_y}\) holds with the sign of the last term as written, for any saved reference \(r\). The last term has population conditional mean zero if \(r\) is the true posterior on the observation set. The JSON reports it for both references.

The empirical CFO information is \(I(Y;S\mid Z)=H(Y\mid Z)-H(Y\mid S,Z)\). The code counts supply both conditional entropies. This is the information available to a Bayes decoder with the sign metadata, independent of which connector is used. Additional diagnostics use the score vector \(Q\) as a covariate. Then \(L(Q)=H(Y)-I(Y;Q)+\mathbb E_Q D_{\rm KL}(P(Y\mid Q)\Vert Q)\). With the fixed feature \(A\) equal to the argmax set of \(Q\), the identity is \(L(Q)=H(Y\mid A)-I(Y;Q\mid A)+\mathbb E_Q D_{\rm KL}(P(Y\mid Q)\Vert Q)\). Argmax resolution reports a coarse, model-dependent predictive association. Exact-float score-vector groups can identify a code or even a pilot row, so their empirical resolution can be high for a poor decoder. Neither diagnostic identifies a phase-slot mechanism.

### Scalar calibration and marginal bias

For logits \(\ell_i\), inverse temperature \(\beta\ge 0\), and empirical weights \(w_i\), let

\[
f(\beta)=\sum_i w_i\big[\log\sum_j e^{\beta \ell_{ij}}-\beta\ell_{iY_i}\big].
\]

Then \(f'=\sum_i w_i(\mathbb E_{q_\beta(\cdot\mid i)}\ell_{ij}-\ell_{iY_i})\) and \(f''=\sum_i w_i\operatorname{Var}_{q_\beta(\cdot\mid i)}\ell_{ij}\ge0\). If \(f'(0)\ge0\), \(\beta^*=0\) and the optimal scalar forecast is the uniform vector. Otherwise solve \(f'(\beta^*)=0\) for an interior optimum when present. The gap at the lock is exactly

\[
f(\beta_d)-f(\beta^*)=
\mathbb E_i D_{\rm KL}(q_{\beta^*}(\cdot\mid i)\Vert q_{\beta_d}(\cdot\mid i))
+(\beta_d-\beta^*)f'(\beta^*).
\]

The boundary derivative term generally remains at \(T^*=\infty\). A finite positive \(T\) preserves each row's ranking, so ECE and NLL can change while accuracy stays fixed. At infinity the four-way tie awards fractional accuracy \(1/4\). Reported oracle fits are label-optimal scalar temperatures, separately for each arm and seed. The JSON also fits a single common raw temperature per arm and cell. ECE uses the same pooled ten top-confidence bins as Stage 3D. ECE is neither proper nor an additive component of log risk.

For any fixed predictor \(q\), define two joints \(P(c,y)=w(c)p_y(c)\) and \(\widetilde Q(c,y)=w(c)q_y(c)\), where \(c\) includes seed when fitting multiple connectors. The chain rule in reverse order yields

\[
\mathbb E_C D_{\rm KL}(p\Vert q)
=D_{\rm KL}(\bar y\Vert\bar q)
+\mathbb E_{Y\sim\bar y}D_{\rm KL}\big(P(C\mid Y)\Vert\widetilde Q(C\mid Y)\big).
\]

The marginal term measures global prior bias. The second term measures conditional allocation error, encompassing misranking, miscalibration, and other errors. A learned intercept adjustment need not realize the marginal term as a standalone reduction in NLL.

### The reported additive path

Let \(q_d\) use the locked temperatures, \(q_*\) use the post-hoc fitted temperatures and \(Pq_*\) denote the arithmetic projection. For the empirical Q3 design the reported identity is

\[
L(q_d)=H(Y\mid S,Z)+[L(q_d)-L(q_*)]+[L(q_*)-L(Pq_*)]
+D_{\rm KL}(\bar y\Vert\overline{Pq_*})
+\mathbb E_Y D_{\rm KL}(P(C\mid Y)\Vert\widetilde Q_{Pq_*}(C\mid Y)).
\]


capture_archives_bytes      : 1807824953
all_selected_download_bytes : 1966425693
selected_extracted_bytes    : 1585139947



16481

m  nu kl_to_nominal          kl_to_moment_matched_gaussian
-  -- -------------          -----------------------------
1   1 infinite                                            
1   2 0.5                    0.1931471805599453           
1   3 0.16666666666666718    0.07213177477483157          
1   5 0.04999999999999871    0.02314355131420842          
1  10 0.011111111111106187   0.005360515657821363         
1  30 0.0011494252873622823  0.0005682183423539877        
1 100 0.00010101010101215557 5.0335853503447E-05          
3   1 infinite                                            
3   2 1.6819456220014448     0.7613871636812806           
3   3 0.5985076962177711     0.31490302054226404          
3   5 0.19260331722469282    0.11203397116732194          
3  10 0.04581355478009419    0.028561768420240163         
3  30 0.005010490790378519   0.003266869955353524         
3 100 0.0004500866857508612  0.0002980639432257348        


{"low_gain_codes":[1,5,9,13],"high_gain_codes":[0,4,8,12],"exact_zero_code":1}
 -2 0 R1_colored_3tap         0.39063 1.15807 0.42104
 -2 0 R2_repeated_GLS         0.37561 1.13212 0.43270
 -2 0 R3_lift_physical_SNR    0.37561 1.12851 0.43381
 -2 0 R3_lift_calibrated_SNR  0.37561 1.00818 0.45530
 -2 0.3 R1_colored_3tap         0.40076 1.13260 0.41562
 -2 0.3 R2_repeated_GLS         0.38879 1.10832 0.42832
 -2 0.3 R3_lift_physical_SNR    0.38879 1.10470 0.42895
 -2 0.3 R3_lift_calibrated_SNR  0.38879 1.05321 0.43604
  2 0 R1_colored_3tap         0.20740 0.56058 0.82939
  2 0 R2_repeated_GLS         0.19983 0.53316 0.85576
  2 0 R3_lift_physical_SNR    0.19983 0.53224 0.85661
  2 0 R3_lift_calibrated_SNR  0.19983 0.53224 0.85661
  2 0.3 R1_colored_3tap         0.23657 0.62692 0.78566
  2 0.3 R2_repeated_GLS         0.22729 0.60787 0.80404
  2 0.3 R3_lift_physical_SNR    0.22729 0.60796 0.80407
  2 0.3 R3_lift_calibrated_SNR  0.32263 0.85943 0.58100
 20 0 R1_colored_3tap         0.00000 0.05958 1.38629
 20 0 R2_repeated_GLS         0.00000 0.05958 1.38629
 20 0 R3_lift_physical_SNR    0.00000 0.05958 1.38629
 20 0 R3_lift_calibrated_SNR  0.19885 0.54274 0.84497
 20 0.3 R1_colored_3tap         0.00000 0.23922 1.38629
 20 0.3 R2_repeated_GLS         0.00000 0.23319 1.38629
 20 0.3 R3_lift_physical_SNR    0.00000 0.23905 1.38629
 20 0.3 R3_lift_calibrated_SNR  0.32544 0.87857 0.56402

kind               value lag_symbols conditional_joint_kl_nats scalar_oracle_uniform_mi_nats_mc nominal_posterior_regre
                                                                                                              t_nats_mc
----               ----- ----------- ------------------------- -------------------------------- -----------------------
coherent_spur       0.26           0       0.10713877981037129                1.011841309761531     0.06302438377556768
cfo_rad_per_symbol  0.02           4      0.010137907816939642               0.8902052804895653    0.009503028948955067
cfo_rad_per_symbol  0.02          25        0.3880371285969559               0.8916411310368753     0.33485006108837345
cfo_rad_per_symbol  0.02         126         5.746670683716246               0.8884646919434117       5.666248303246695
student_t_nu           1           0                  infinite               0.5870331042533227      1.0886499948145008
student_t_nu           3           0       0.16666666666666718               0.7491312325380528     0.06877750150452314


```

### Event #47: `item.completed` — `agent_message` (`item_44`)

The CPU replay passed the first 168 mapping cells and 32 nuisance cases. In the clean 20 dB surrogate, preserving the measured SNR gave 0 errors in 8,192 symbols with the frozen table, while forcing the input to 2 dB gave 19.9% SER. I’ll use the SNR-preserving lift as the primary mapping and keep forced calibration as an ablation.

### Event #48: `item.completed` — `file_change` (`item_45`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M4\\verify_m4.py",
    "kind": "update"
  }
]
```

### Event #49: `item.completed` — `file_change` (`item_46`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M4\\THEOREM_M4.md",
    "kind": "add"
  }
]
```

### Event #50: `item.completed` — `web_search` (`exec-7926761d-d1d0-4bbf-8e8c-659704400bc5`)

```json
{
  "id": "exec-7926761d-d1d0-4bbf-8e8c-659704400bc5",
  "type": "web_search",
  "query": "'physical'",
  "action": {
    "type": "find_in_page",
    "pattern": "physical"
  },
  "results": [
    {
      "type": "text_result",
      "domain": "modal.com",
      "ref_id": "turn2view0",
      "snippet": "Total lines: 84",
      "thumbnail_url": "https://images.openai.com/static-rsc-1/k3YouBoJjgb8qg0V3O5jO5L6PgYIN_kEzg26DscRe1gxOheuV5BSafnD6lTF0MHXslYfDhWiqxZGUC7hIaWJMfIDEQj1Fcviw-AdYls0EP9OCPxIKHzbaLWcdj0CKcl31G1EpmK17P4RPD-93144at2M8WtI1S9x0uunhYCX5Eo",
      "title": "Configuring CPU, memory, and disk | Modal Docs",
      "url": "https://modal.com/docs/guide/resources"
    },
    {
      "type": "text_result",
      "domain": "modal.com",
      "ref_id": "turn2view1",
      "snippet": "Total lines: 84",
      "thumbnail_url": "https://images.openai.com/static-rsc-1/kx0cXnVARa6ANQR64ara3-rKyE4oWdRVdfqPlO3jXdQokZEmyA3qkQKArAf0FUCXO868CJ15OYjjk_zfQ2WB3uGh35Co_LPE57HmZZejlBDMjzwP2FT3mUstEtoOuK9bT9ijMl2JdQVDm6ON-Ld4CmpcBBhvrjCKjhQzV0cdsKg",
      "title": "Configuring CPU, memory, and disk | Modal Docs",
      "url": "https://modal.com/docs/guide/resources"
    },
    {
      "type": "text_result",
      "domain": "modal.com",
      "ref_id": "turn2view2",
      "snippet": "Total lines: 392",
      "thumbnail_url": "https://images.openai.com/static-rsc-1/BF57dEbo-HHD4f9_gh7X-5_-DHR0zvJctAQnI10QRFbhcajS9bV5vqFWYk4MhddEr9RkspOqmx7x8bxDDXIbT94s0mYHRH07ziv0_qXunvPzotN4Jm_M9pAhpLdSX5ip",
      "title": "Volumes | Modal Docs",
      "url": "https://modal.com/docs/guide/volumes"
    }
  ]
}
```

### Event #51: `item.completed` — `command_execution` (`item_47`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$env:UV_CACHE_DIR=(Resolve-Path '"'research/stage4_math_and_design/M4/.uv-m4').Path; "'$env:TEMP=$env:UV_CACHE_DIR; $env:TMP=$env:UV_CACHE_DIR; $env:UV_OFFLINE='"'1'; "'$env:UV_NO_PROJECT='"'1'; "'$env:UV_LINK_MODE='"'copy'; "'$env:OPENBLAS_NUM_THREADS='"'1'; uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B research/stage4_math_and_design/M4/verify_m4.py --check-http"
```

**Aggregated Output:**

```text
Verified frozen s17 identity and pinned Python/NumPy/SciPy
Verified static splits, historical manifests and Modal arithmetic
PASS 168 mapping cells, 32 nuisance cells, results_m4.json
```

### Event #52: `item.completed` — `file_change` (`item_49`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M4\\verify_m4.py",
    "kind": "update"
  }
]
```

### Event #53: `item.completed` — `file_change` (`item_50`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M4\\THEOREM_M4.md",
    "kind": "update"
  }
]
```

### Event #54: `item.completed` — `command_execution` (`item_51`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$env:UV_CACHE_DIR=(Resolve-Path '"'research/stage4_math_and_design/M4/.uv-m4').Path; "'$env:TEMP=$env:UV_CACHE_DIR; $env:TMP=$env:UV_CACHE_DIR; $env:UV_OFFLINE='"'1'; "'$env:UV_NO_PROJECT='"'1'; "'$env:UV_LINK_MODE='"'copy'; "'$env:OPENBLAS_NUM_THREADS='"'1'; uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B research/stage4_math_and_design/M4/verify_m4.py --check-http"
```

**Aggregated Output:**

```text
Verified frozen s17 identity and pinned Python/NumPy/SciPy
Verified static splits, historical manifests and Modal arithmetic
PASS 168 mapping cells, 32 nuisance cells, results_m4.json
```

### Event #55: `item.completed` — `command_execution` (`item_52`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n '"'^# Track|''^## [123]|''^'"\\s*[-*].*(M4|Stage4\\+Ref|calibration)' research/stage4_math_and_design/discussions/M1/prompt_r1.md research/stage4_math_and_design/discussions/M2/prompt_r1.md research/stage4_math_and_design/discussions/M3/prompt_r1.md"
```

**Aggregated Output:**

```text
research/stage4_math_and_design/discussions/M3/prompt_r1.md:1:# Track M3 (Round 1): Minimal Bayesian Sufficiency for Variable Pilot Multisets (Q2 / Q3_after_pilot) & Idempotent Pooling
research/stage4_math_and_design/discussions/M3/prompt_r1.md:14:## 1. Grounded Empirical Context (Read These Files Read-Only First)
research/stage4_math_and_design/discussions/M3/prompt_r1.md:40:## 2. Mathematical Questions You Must Answer Rigorously
research/stage4_math_and_design/discussions/M3/prompt_r1.md:69:## 3. Deliverables Required from Track M3
research/stage4_math_and_design/discussions/M2/prompt_r1.md:1:# Track M2 (Round 1): Information-Theoretic Bregman/KL Decomposition, Calibration Forensics & Stage 4 Statistical Gate
research/stage4_math_and_design/discussions/M2/prompt_r1.md:14:## 1. Grounded Empirical Context (Read These Files Read-Only First)
research/stage4_math_and_design/discussions/M2/prompt_r1.md:31:## 2. Mathematical & Forensic Questions You Must Answer Rigorously
research/stage4_math_and_design/discussions/M2/prompt_r1.md:53:## 3. Deliverables Required from Track M2
research/stage4_math_and_design/discussions/M1/prompt_r1.md:1:# Track M1 (Round 1): Group-Theoretic Phase-Coupled Equivariant Architecture & Extrapolation Theory
research/stage4_math_and_design/discussions/M1/prompt_r1.md:15:## 1. Grounded Empirical Context (Read These Files Read-Only First)
research/stage4_math_and_design/discussions/M1/prompt_r1.md:35:## 2. Mathematical Questions You Must Answer Rigorously
research/stage4_math_and_design/discussions/M1/prompt_r1.md:48:   - Explicitly specify clean interfaces for **Track M3 (Q2/Q3_after_pilot `RefSufficientPool`)** and **Track M4 (Real Pluto/CTTC CFO+SNR inputs)**!
research/stage4_math_and_design/discussions/M1/prompt_r1.md:61:## 3. Deliverables Required from Track M1
```

### Event #56: `item.completed` — `command_execution` (`item_53`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command 'git status --short --untracked-files=normal; git rev-parse HEAD; Get-ChildItem -Force research/stage4_math_and_design/M4 | Select-Object Name,Length'
```

**Aggregated Output:**

```text
?? research/stage4_math_and_design/
387e7bc749150a69fb317b3f1351ad0d07b3bfad

Name            Length 
----            ------ 
.uv-m4                 
results_m4.json 1475412
THEOREM_M4.md   46422  
verify_m4.py    34558  


```

### Event #57: `item.completed` — `file_change` (`item_54`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M4\\verify_m4.py",
    "kind": "update"
  }
]
```

### Event #58: `item.completed` — `file_change` (`item_55`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M4\\THEOREM_M4.md",
    "kind": "update"
  }
]
```

### Event #59: `item.completed` — `command_execution` (`item_56`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$env:UV_CACHE_DIR=(Resolve-Path '"'research/stage4_math_and_design/M4/.uv-m4').Path; "'$env:TEMP=$env:UV_CACHE_DIR; $env:TMP=$env:UV_CACHE_DIR; $env:UV_OFFLINE='"'1'; "'$env:UV_NO_PROJECT='"'1'; "'$env:UV_LINK_MODE='"'copy'; "'$env:OPENBLAS_NUM_THREADS='"'1'; uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B research/stage4_math_and_design/M4/verify_m4.py --check-http"
```

**Aggregated Output:**

```text
Verified frozen s17 identity and pinned Python/NumPy/SciPy
Verified static splits, historical manifests and Modal arithmetic
PASS 168 mapping cells, 32 nuisance cells, results_m4.json
```

### Event #60: `item.completed` — `command_execution` (`item_57`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$doc=Get-Content -Raw research/stage4_math_and_design/M4/results_m4.json | ConvertFrom-Json; $doc.dataset_manifests.live_http | ConvertTo-Json -Compress; $doc.frozen_encoder_sweep.ray_check | ConvertTo-Json -Compress; $doc.nuisance_curves.random_bounded_gain_mixture | Select-Object mixture_kl_nats_mc,I_scale_observation_given_symbol_nats_mc,hidden_scale_uniform_mi_nats_mc,known_scale_uniform_mi_nats_mc | ConvertTo-Json -Compress; $doc.modal_budget | Select-Object stage4_total_usd_cap,headroom_to_historical_remaining_usd | ConvertTo-Json -Compress; $doc | Select-Object status,python,numpy,scipy | ConvertTo-Json -Compress; rg -n '"'"'^[#]'"' research/stage4_math_and_design/M4/THEOREM_M4.md"
```

**Aggregated Output:**

```text
{"mendeley_files":{"metadata_reachable":false,"detail":"\u003curlopen error [WinError 10061] ����Ŀ�����������ܾ����޷����ӡ�\u003e"},"zenodo_files":{"metadata_reachable":false,"detail":"\u003curlopen error [WinError 10061] ����Ŀ�����������ܾ����޷����ӡ�\u003e"}}
{"low_gain_codes":[1,5,9,13],"high_gain_codes":[0,4,8,12],"certified_low_ray_gain_upper":0.03205798037553048,"certified_high_ray_gain_lower":21.895815031309443,"exact_zero_code":1}
{"mixture_kl_nats_mc":0.18551057519218767,"I_scale_observation_given_symbol_nats_mc":0.2272181945364308,"hidden_scale_uniform_mi_nats_mc":0.8732857257272549,"known_scale_uniform_mi_nats_mc":0.889764001452556}
{"stage4_total_usd_cap":"8.44073416","headroom_to_historical_remaining_usd":"4.43926584"}
{"status":"pass","python":"3.12.7","numpy":"2.3.5","scipy":"1.17.0"}
1:# Track M4 · Real SDR bridge for Stage 4
7:## 1. Observation laws and losses
9:### 1.1 The three representations
29:### 1.2 Conditional KL and perturbation terms
74:### 1.3 Student scale, Rayleigh scale, and the frozen nonlinearity
112:## 2. Hardware benchmark and causality contract
114:### 2.1 Candidate rows and time order
129:### 2.2 What zero leakage requires and what was verified
145:### 2.3 Same-row ablations and pooling
158:## 3. Frozen compute envelope and dispatch
160:### 3.1 Dataset inputs and handoffs
172:### 3.2 Model and scored workload
180:### 3.3 Budget arithmetic and zero-bill gate
202:## 4. CPU replay and interpretation
```

### Event #61: `item.completed` — `file_change` (`item_58`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M4\\THEOREM_M4.md",
    "kind": "update"
  }
]
```

### Event #62: `item.completed` — `file_change` (`item_59`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M4\\reply_r1.md",
    "kind": "add"
  }
]
```

### Event #63: `item.completed` — `agent_message` (`item_60`)

The final replay passed with the frozen weight hash, archived split checks, and the $8.44073416 budget calculation. The remaining real-data gate is causal acquisition timing. The report records which legacy frontend inputs must be excluded and what the live provenance audit needs to establish.

### Event #64: `item.completed` — `command_execution` (`item_61`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "Resolve-Path -LiteralPath 'research/stage4_math_and_design/M4','research/stage4_math_and_design/M4/.uv-m4'; Get-ChildItem -LiteralPath 'research/stage4_math_and_design/M4/.uv-m4' -Force -Recurse -Attributes ReparsePoint | Select-Object -ExpandProperty FullName; git rev-parse HEAD; git --no-optional-locks status --short --untracked-files=normal"
```

**Aggregated Output:**

```text

387e7bc749150a69fb317b3f1351ad0d07b3bfad
?? research/stage4_math_and_design/
Path                                                                                                       
----                                                                                                       
C:\Users\������\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M4       
C:\Users\������\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M4\.uv-m4


```

### Event #65: `item.completed` — `command_execution` (`item_62`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "uv cache clean --help; Get-ChildItem -Force -Name 'research/stage4_math_and_design/M4/.uv-m4' "
```

**Aggregated Output:**

```text
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
      --color <COLOR_CHOICE>
          Control the use of color in output [possible values: auto, always, never]
      --system-certs
          Whether to load TLS certificates from the platform's native certificate store [env:
          UV_SYSTEM_CERTS=]
      --offline
          Disable network access [env: UV_OFFLINE=]
      --allow-insecure-host <ALLOW_INSECURE_HOST>
          Allow insecure connections to a host [env: UV_INSECURE_HOST=]
      --no-progress
          Hide all progress outputs [env: UV_NO_PROGRESS=]
      --directory <DIRECTORY>
          Change to the given directory prior to running the command [env: UV_WORKING_DIR=]
      --project <PROJECT>
          Discover a project in the given directory [env: UV_PROJECT=]
      --config-file <CONFIG_FILE>
          The path to a `uv.toml` file to use for configuration [env: UV_CONFIG_FILE=]
      --no-config
          Avoid discovering configuration files (`pyproject.toml`, `uv.toml`) [env: UV_NO_CONFIG=]
  -h, --help
          Display the concise help for this command
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

### Event #66: `item.completed` — `command_execution` (`item_63`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "uv cache clean --cache-dir 'C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M4/.uv-m4' --offline"
```

**Aggregated Output:**

```text
Clearing cache at: research/stage4_math_and_design/M4/.uv-m4
Removed 4948 files (295.5MiB)
```

### Event #67: `item.completed` — `command_execution` (`item_65`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$r=Get-Content -Encoding utf8 -Raw research/stage4_math_and_design/M4/results_m4.json | ConvertFrom-Json; $r.status; $r.frozen_encoder_sweep.cells.Count; $r.nuisance_curves.conditional_curves.Count; $r.modal_budget.stage4_total_usd_cap; $r.dataset_manifests.total_download_bytes; $r.index_audit.cttc.counts | ConvertTo-Json -Compress'
```

**Aggregated Output:**

```text
pass
168
32
8.44073416
2789188515
{"train_target":1440,"train_ref":576,"dev_target":864,"dev_ref":576,"test_target":1440,"test_ref":576}
```

### Event #68: `item.completed` — `command_execution` (`item_64`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command 'Get-ChildItem -Force research/stage4_math_and_design/M4 | Select-Object Name,Length; git rev-parse HEAD; git --no-optional-locks status --short --untracked-files=all -- research/stage4_math_and_design/M4'
```

**Aggregated Output:**

```text

387e7bc749150a69fb317b3f1351ad0d07b3bfad
?? research/stage4_math_and_design/M4/THEOREM_M4.md
?? research/stage4_math_and_design/M4/reply_r1.md
?? research/stage4_math_and_design/M4/results_m4.json
?? research/stage4_math_and_design/M4/verify_m4.py
Name             Length
----             ------
reply_r1.md        5588
results_m4.json 1475582
THEOREM_M4.md     47645
verify_m4.py      36179


```

### Event #69: `item.completed` — `agent_message` (`item_66`)

M4 Round 1 is complete. The pinned `uv` replay passed **168 mapping cases and 32 nuisance cases** using the frozen `equivariant_b8_s17` weights.

- **Recommended mapping.** Use the SNR-preserving R3 lift. It preserves the measured scalar and makes the three-tap joint KL equal to the scalar joint KL. In the clean 20 dB surrogate, it scored **0 errors in 8,192 symbols**. Forcing the input to 2 dB gave **19.89% SER**.
- **Misspecification.** The exact scalar Gaussian KL is \(\lambda-1-\log\lambda+|\mu_s-s|^2/V_0\). A residual CFO of \(0.020\) rad/symbol costs **0.010138, 0.388037 and 5.746671 nats** at lags 4, 25 and 126. An independent Rayleigh scale denominator gives infinite forward Gaussian KL. The scalar Student case with three complex residual degrees of freedom gives **0.166667 nats**, or **0.072132 nats** after variance matching.
- **Real-data protocol.** Lock W1/W2 before releasing one or two past pilots. Use the next Pluto copy and the next CTTC loop at the same source index for Q3. The verifier confirmed the archived split and local interpolation geometry for **2,240 Pluto frames and 1,440 CTTC packets**, including **720 CTTC Q3 pairs**. Live causal timing and raw-sample provenance remain the real-data qualification gate. The replay used synthetic IQ.
- **Modal.** The planned cap is **$8.44073416**, including **10,274 A100-80GB seconds**, CPU/RAM and retry reserve. That leaves **$4.43926584** against the historical $12.88 balance. Dispatch depends on dev throughput, live credits and a $0.00 net spend guard. Modal use in this round was zero. [Modal pricing](https://modal.com/pricing) · [Spend limits](https://modal.com/docs/guide/budgets)

Round 2 needs three interface decisions. **M1** should lock the receive-gauge action and the CFO uncertainty input for one canonical host score. **M2** should lock the paired NLL contrasts and cluster inference. **M3** should specify how the pool uses correlated pilot innovations without double counting evidence already in the causal state.

[Theorem and protocol](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M4/THEOREM_M4.md) · [CPU verifier](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M4/verify_m4.py) · [Results JSON](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M4/results_m4.json) · [Round 1 handoff](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M4/reply_r1.md)



### Event #70: `turn.completed`

- Usage: `{"input_tokens": 7860310, "cached_input_tokens": 7697536, "cache_write_input_tokens": 0, "output_tokens": 51585, "reasoning_output_tokens": 10353}`


## Round 1 — Final Output (`reply_r1.md`)

```markdown
M4 Round 1 is complete. The pinned `uv` replay passed **168 mapping cases and 32 nuisance cases** using the frozen `equivariant_b8_s17` weights.

- **Recommended mapping.** Use the SNR-preserving R3 lift. It preserves the measured scalar and makes the three-tap joint KL equal to the scalar joint KL. In the clean 20 dB surrogate, it scored **0 errors in 8,192 symbols**. Forcing the input to 2 dB gave **19.89% SER**.
- **Misspecification.** The exact scalar Gaussian KL is \(\lambda-1-\log\lambda+|\mu_s-s|^2/V_0\). A residual CFO of \(0.020\) rad/symbol costs **0.010138, 0.388037 and 5.746671 nats** at lags 4, 25 and 126. An independent Rayleigh scale denominator gives infinite forward Gaussian KL. The scalar Student case with three complex residual degrees of freedom gives **0.166667 nats**, or **0.072132 nats** after variance matching.
- **Real-data protocol.** Lock W1/W2 before releasing one or two past pilots. Use the next Pluto copy and the next CTTC loop at the same source index for Q3. The verifier confirmed the archived split and local interpolation geometry for **2,240 Pluto frames and 1,440 CTTC packets**, including **720 CTTC Q3 pairs**. Live causal timing and raw-sample provenance remain the real-data qualification gate. The replay used synthetic IQ.
- **Modal.** The planned cap is **$8.44073416**, including **10,274 A100-80GB seconds**, CPU/RAM and retry reserve. That leaves **$4.43926584** against the historical $12.88 balance. Dispatch depends on dev throughput, live credits and a $0.00 net spend guard. Modal use in this round was zero. [Modal pricing](https://modal.com/pricing) · [Spend limits](https://modal.com/docs/guide/budgets)

Round 2 needs three interface decisions. **M1** should lock the receive-gauge action and the CFO uncertainty input for one canonical host score. **M2** should lock the paired NLL contrasts and cluster inference. **M3** should specify how the pool uses correlated pilot innovations without double counting evidence already in the causal state.

[Theorem and protocol](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M4/THEOREM_M4.md) · [CPU verifier](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M4/verify_m4.py) · [Results JSON](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M4/results_m4.json) · [Round 1 handoff](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M4/reply_r1.md)

```


---

## Round 2 — Cross-Review & Convergence Prompt (`prompt_r2.md`)

```markdown
# Track M4 (Round 2): Unified A100-Safe Modal Budget, Dual `s17`+`16-bit` Causal Frontend & Full M1/M2/M3 Real-Data Lock

You are resuming persistent Codex session **Track M4** (`01a21948-980b-7413-a967-3074c32be83a`) in `C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4`. Keep edits STRICTLY inside `research/stage4_math_and_design/M4/` and run Python ONLY via `uv run --python 3.12 ...`.

## 1. Cross-Track Tensions & Contracts to Reconcile (From M1, M2, M3, M5, M6)
1. **Reconciling Modal GPU Schedule Between M4 (`$8.44` pure A100-80GB) and M6 (`$9.50` mixing L4/A10G + 173-step pilot)**:
   - Look closely at why you and M6 differed slightly:
     - M6 budgeted `L4` (`24 GB`, `$0.7992/h`) for Pluto & CTTC forward eval (`4,000s` total) + A100-80GB sequential 1-seed pilot (`<= 173/960` steps at $t=0.18$, then continue up to `3` seeds).
     - You correctly cited Stage 3D's MEASURED peaks on A100-80GB (`52,972.5 MB` train at microbatch 16; **`45,409.9 MB` eval at batch 64**!) and showed that running pure **A100-80GB** STILL costs only **`$8.4407`** if training `1` seed across `4` arms (`384` steps/arm)!
   - Wait — **M2's Statistical Reality Check** interacts critically with BOTH:
     - M2 proved `1` seed or `3` seeds CANNOT establish population-of-seeds generalization (`>= 10` seeds needed for that!), so Stage 4's real-data + synthetic GPU budget MUST explicitly state a **Fixed-Connector Conditional Scope** OR use `1` seed per each of the `4` matched arms (`Zero`, `Naive-Additive`, `Host-Pure Equivariant+RefPool`, `Physical+HostResidual` -> M1's converged 4-arm ablation!) + optional `2` extra seeds ONLY for the winning M1 `Host-Pure / Physical-Residual` pair if Stage-1 dev pilot clears M6's Tier-2 gate!
   - **Resolve this cleanly with exact USD arithmetic (`<= $9.50` cap inside `$12.88` remaining credits)**: Put A100-80GB (`$2.4984/h + cpu/ram = $0.00074684/s` or M6's rate check — wait: standardize the EXACT Modal pricing formula!) as the verified VRAM-safe default (`no unverified L4 assumption unless a $0.10 micro-batch parity smoke proves <= 20 GB!`), reconcile M6's `$0.49` 1-seed pilot gate cleanly BEFORE spending full arm budgets, and check how zero-shot `f708_snr10` holdout fits inside the same row count!
2. **Unified Causal Dual-Cache Extraction (`4-bit s17` + `16-bit canonical` M5 Packet) in ONE Zero-GPU CPU Pass**:
   - M5 proved `4-bit s17` saturates along rays (which you ALSO proved independently — `local` saturates to `[0,4,8,12]` above gain `21.90` and gives constant code below `0.032`!), causing Stage 5–6 NLL loss (`0.3119 nats`) and hiding SNR changes across CTTC (`-3..+24 dB`) and Pluto fading (`>10.6 dB`).
   - Since your CPU causal frontend ALREADY computes R3 whitened complex `T`, residual energy `E`, normalized radial SNR `r`, and `equivariant_b8_s17(Y_R3)`, extracting BOTH **`4-bit s17` codes** AND **M5's `16-bit canonical packet` `[2b turn | 2b legacy_inner | 6b canon_angle | 4b log_radial | 2b resid_quality]`** costs **`0` extra GPU time and `< 2 MB` storage**! Incorporate this dual export cleanly into Stage 4's CPU shard schema AND check CPU physical ceilings (`4-bit s17` vs `16-bit canonical`) before launching GPU!
3. **Honoring M1 + M3Exact Interfaces on Pluto & CTTC**:
   - M1 expects exact `PhaseContext`: signed `omega_hat` in `T(t) ~ e^{+i omega (t - t1)}`, causal prior CFO variance `v_omega`, `diffusion_var_per_unit`, calibrated SNR, and explicit distinction between **receive-gauge turn** `r_rx` (Q1/Q2 invariant, Q3 RX sector shift `+r_rx`) and TX relabel.
   - M3 expects `RefSufficientPool` inputs with **NO DOUBLE-COUNTING** of pilots already folded into $\mathcal{F}^0$ (`0:13` sync Pluto / `0:6` sync CTTC excluded from revealed set; reveal ONLY `15, 18` Pluto & `126, 127` CTTC with stable salted `acquisition_id`, exact causal timestamps, noise covariance $K_{\text{ref}}$, and shared CFO coupling `(omega_hat, v_omega)`!).
   - M2 requires all `9` CTTC captures explicitly represented in session-level sensitivity alongside the `72` loops / `36` Q3 loop-pairs, and zero-exclusion coverage accounting!

## 2. Specific Round-2 Convergence Tasks for Track M4
Update `research/stage4_math_and_design/M4/THEOREM_M4.md`, `verify_m4.py`, and `results_m4.json`:
1. Lock the **exact `PhaseContext` + `RefPool` + `Dual 4-bit s17 / 16-bit canonical` CPU Shard Contract** for Pluto (`7,096` frames) and CTTC (`9` captures), explicitly checking causal filtration non-double-counting and receive-gauge vs TX-relabel invariance!
2. Reconcile the **Modal A100-80GB Tier-2 Pilot-Gated Schedule (`<= $9.50` cap, `$0.00` net guard)** combining M1's 4-arm ablation (`Zero`, `Naive-Additive`, `Equivariant-Host-Pure`, `Equivariant-Physical+Residual`), M6's dev pilot futility gate BEFORE full continuation, and `f708_snr10` held-out session transfer! (Explain precisely why M4+M6 pricing rates differed slightly if CPU/RAM sizing differed: `2c/12 GiB` vs `2c/16 GiB`, and lock one exact Modal pricing table!)
3. Extend `verify_m4.py` concisely to test: (i) the unified **`4-bit s17` vs `16-bit canonical` CPU ceiling gap** under Pluto-like fading (`10 dB` dynamic range) & CTTC-like SNR (`-3, +10, +20 dB`) + `0.26` spur; (ii) exact receive-gauge transformation law & non-double-counting causal ledger audit on simulated Pluto/CTTC splits; and (iii) exact reconciled Modal ledger arithmetic. Run via `uv run` and report in `reply_r2.md`.
```


## Round 2 — Resumed Execution & Reasoning Trace (`round2_events.jsonl`)

### Event #1: `thread.started` (thread_id=`01a21948-980b-7413-a967-3074c32be83a`)

### Event #2: `turn.started`

### Event #3: `item.completed` — `agent_message` (`item_0`)

I’ll read the cross-track contracts and the Round 1 artifacts first, then reconcile the A100 budget, lock the dual-cache schema, and rerun the CPU verifier.

### Event #4: `item.completed` — `command_execution` (`item_1`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n -i 'EM.FloatFormer|stage.?4|fixed.connector' 'C:/Users/李则徐/.codex/memories/MEMORY.md'"
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
```

### Event #5: `item.completed` — `command_execution` (`item_2`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "git --no-optional-locks status --short; rg --files --hidden research/stage4_math_and_design -g '*.md' -g '*.py' -g '*results*.json' -g '"'!**/.uv*/**'"'; rg --files -g AGENTS.md -g '"'!**/.uv*/**'"'; uv --version"
```

**Aggregated Output:**

```text
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
research/stage4_math_and_design\M4\verify_m4.py
research/stage4_math_and_design\M4\THEOREM_M4.md
research/stage4_math_and_design\M4\results_m4.json
research/stage4_math_and_design\M4\reply_r1.md
research/stage4_math_and_design\M3\verify_m3.py
research/stage4_math_and_design\M3\THEOREM_M3.md
research/stage4_math_and_design\M3\results_m3.json
research/stage4_math_and_design\M3\reply_r1.md
research/stage4_math_and_design\M1\verify_m1.py
research/stage4_math_and_design\M1\THEOREM_M1.md
research/stage4_math_and_design\M1\results_m1.json
research/stage4_math_and_design\M2\verify_m2.py
research/stage4_math_and_design\M2\THEOREM_M2.md
research/stage4_math_and_design\M2\results_m2.json
research/stage4_math_and_design\M2\reply_r1.md
research/stage4_math_and_design\discussions\M1\reply_r1.md
research/stage4_math_and_design\discussions\M1\prompt_r2.md
research/stage4_math_and_design\discussions\M1\prompt_r1.md
research/stage4_math_and_design\discussions\M6\reply_r1.md
research/stage4_math_and_design\discussions\M6\prompt_r2.md
research/stage4_math_and_design\discussions\M6\prompt_r1.md
research/stage4_math_and_design\discussions\M3\reply_r1.md
research/stage4_math_and_design\discussions\M3\prompt_r2.md
research/stage4_math_and_design\discussions\M3\prompt_r1.md
research/stage4_math_and_design\discussions\M2\reply_r1.md
research/stage4_math_and_design\discussions\M2\prompt_r2.md
research/stage4_math_and_design\discussions\M2\prompt_r1.md
research/stage4_math_and_design\discussions\M4\reply_r1.md
research/stage4_math_and_design\discussions\M5\reply_r1.md
research/stage4_math_and_design\discussions\M4\prompt_r2.md
research/stage4_math_and_design\discussions\M5\prompt_r2.md
research/stage4_math_and_design\discussions\M4\prompt_r1.md
research/stage4_math_and_design\discussions\M5\prompt_r1.md
uv 0.11.21 (5aa65dd7a 2026-06-11 x86_64-pc-windows-msvc)
```

### Event #6: `item.completed` — `command_execution` (`item_3`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n 'PhaseContext|omega_hat|diffusion_var|ref_pool|Physical|Host.Pure|action|relabel' research/stage4_math_and_design/M1/THEOREM_M1.md research/stage4_math_and_design/M1/verify_m1.py; rg -n 'sufficient|acquisition_id|K_ref|conditional|variance|circular' research/stage4_math_and_design/M3/reply_r1.md research/stage4_math_and_design/M3/THEOREM_M3.md"
```

**Aggregated Output:**

```text
research/stage4_math_and_design/M1/THEOREM_M1.md:5:Stage 4 makes the quarter-turn output action an explicit permutation and transports evidence by the signed physical phase before sector readout. The permutation law holds independently of the frozen host. The transport accounts for the Stage 3D direct-table recovery near a quarter turn. The Qwen failure is empirical. Separate token positions can interact, and neither a universal below-chance bound for extrapolation nor a fixed sample-complexity penalty follows from the architecture.
research/stage4_math_and_design/M1/THEOREM_M1.md:19:### Theorem 1 — Encoder and conditional choice actions
research/stage4_math_and_design/M1/THEOREM_M1.md:33:For the joint law, relabel \(X_1\mapsto X_1+r,\ X_2\mapsto X_2+s\), and rotate noises in windows 1, 2, and the repeat window by their designated turns. Keep \(G,\phi\), and the pilots fixed. The uniform source and circular noise laws preserve the joint measure. The three nonuniform label shifts then follow by subtraction, by reading \(X_1\), or by taking the sector. This yields the conditional identities on the support. For the fresh symbol, condition further on \(G\) and the entire phase path. The new symbol and noise are still independent, and mixing the four roots makes its received law quarter-turn invariant. Marginalization preserves the uniform sector law. \(\square\)
research/stage4_math_and_design/M1/THEOREM_M1.md:35:To allow a common pilot rotation \(r_p\), relabel \(G\mapsto G+r_p\), \(X_1\mapsto X_1+r-r_p\) and \(X_2\mapsto X_2+s-r_p\). Rotate all shared pilots by \(i^{r_p}\) and the repeat window by \(i^r\). Q2 then shifts by \(r-r_p\), while Q1 and the repeat tasks retain their shifts. In particular, Q2 is invariant under a joint receiver-coordinate rotation \(r=s=r_p\). A common pilot turn respects the shared channel and phase path.
research/stage4_math_and_design/M1/THEOREM_M1.md:62:The quarter-turn actions commute with continuous translations wherever both are realized as rotations of the underlying continuous variables. Conjugation therefore gives the trivial \(S^1\to\operatorname{Aut}(H)\) action. The corresponding semidirect product is a direct product, possibly with a kernel if two factors rotate the same variable. Any continuous action \(S^1\to\operatorname{Aut}(H)\) is trivial because the domain is connected and the codomain is finite. The same reasoning precludes a nontrivial continuous action on the finite codes or choices by permutations. A CFO change is an \(\mathbb R\) flow on the joint edge-phase torus, with edge phases \(\Delta\omega\,d\). It has a common frequency period only if the active durations are commensurate. [Group-action background](https://proceedings.mlr.press/v48/cohenc16.html)
research/stage4_math_and_design/M1/THEOREM_M1.md:127:Work F gives orientation gain \(+0.00982\) and mean swapped-minus-matched NLL \(+0.01837\) across phase seeds. The phase seed-averaged table has mean-square metadata effect \(1.85378\cdot10^{-4}\), waveform effect \(1.11977\cdot10^{-3}\), and interaction \(6.36253\cdot10^{-5}\). The physical calibration has metadata effect \(0.201603\), over a thousand times the host's mean-square value. Work G has rot01 and rot11 mean maximum coordinate differences of \(0.05914\) and \(0.12627\), with worst tested discrepancy \(0.48167\) at rot01. These measurements establish weak response in the correct signed direction and broken quotient equivariance for the trained host. [Stage 3D summary](../../stage3d_qwen_signed_phase/results/summary_master.json)
research/stage4_math_and_design/M1/THEOREM_M1.md:135:Separate token emission does not satisfy the theorem's final-logit hypothesis. A frozen attention head can form \(q(z)^\mathsf Tk(\phi)\) with rank up to the projected key dimension. A fixed projection using two coordinates can form \(x(z)\cos\phi+y(z)\sin\phi\), and a second gives the other component of a complex rotation. The host also has nonlinearities and a numerical text stream. For the released fit, the output architecture enforces neither an equivariant code action nor a continuous transport law. The fit's measured extrapolation failure does not establish an expressivity limit of frozen hosts. [Scaled dot-product attention](https://arxiv.org/abs/1706.03762)
research/stage4_math_and_design/M1/THEOREM_M1.md:143:| Physical `cal_ref` | \(0.405035,\ 0.401909,\ 6.20094\cdot10^{-5}\) | 0.898335 | 1.99244 | 2.00149 |
research/stage4_math_and_design/M1/THEOREM_M1.md:162:1. **Canonicalize the inputs.** Accept integer `codes[B,2]`, a task enum, `PhaseContext`, and optional M3 evidence. Compute \((u,v)\) and the task's output shift. The branch sees only canonical inputs, invariant context, and fixed-frame pilot evidence. Return exact uniform probabilities immediately for `q3_indep`, and for `q2` when there is no usable shared pilot.
research/stage4_math_and_design/M1/THEOREM_M1.md:176:### M4 interface — `PhaseContext` for Pluto and CTTC
research/stage4_math_and_design/M1/THEOREM_M1.md:178:- Supply `t1`, `t2`, `t3` and per-pilot timestamps in a common time unit and clock basis. Supply signed `omega_rad_per_unit` under \(T(t)\propto e^{+i\omega(t-t_1)}\), `diffusion_var_per_unit`, `snr_linear` with matched-filter normalization, and the frame and tracking state. Synthetic mapping is `t1=0`, `t2=h`, `t3=h+ell` and variance rate \(\sigma_\phi^2\). CFO in Hz uses \(\omega=2\pi f_{\rm CFO}\) with time in seconds. If the front end already de-rotates I/Q, pass the residual under the same sign convention.
research/stage4_math_and_design/M1/THEOREM_M1.md:190:The verifier uses the production loader and fingerprint, regenerates the paired mothers, and matches all blind seeds, code pairs, and repeat targets to the archive. It uses 20,480 signed-CFO training rows, 16 fixed epochs, three optimizer seeds, and a single shared frozen MLP architecture with identical weights and fixed text embedding. A has 6,936 trainable parameters and B has 6,520. B uses a frozen physical prior while A learns from the inputs only. Scores use fractional ties. Paired intervals average signs and rotations within each 64-mother block. The Q1, Q2, and pilot-task tests exercise the quotient lift with synthetic canonical scores. Trainable end-to-end metrics are for Q3 repeat, with Q3 independent evaluated as a uniform control.
research/stage4_math_and_design/M1/verify_m1.py:5:from fractions import Fraction
research/stage4_math_and_design/M1/verify_m1.py:225:    """Return the maximum probability defect over all codes and all 16 actions."""
research/stage4_math_and_design/M1/verify_m1.py:318:    fraction = e[e > 0] / total if total else np.empty(0)
research/stage4_math_and_design/M1/verify_m1.py:325:        "effective_rank_entropy_of_gram_energy": float(np.exp(-np.dot(fraction, np.log(fraction))))
research/stage4_math_and_design/M1/verify_m1.py:339:        return [Fraction(float(p)).limit_denominator(2 * (65_536 + 2)) for p in z]
research/stage4_math_and_design/M3/THEOREM_M3.md:5:The C4 generator sets the continuous drift to zero at the first window. It draws independent Gaussian increments with mean \(\omega\,dt\) and variance \(\sigma_\phi^2dt\) on the grid \(\{0,h,h+\ell,t_p\}\). Its two valid pilots share the same physical phase and have independent complex AWGN. At \(h=\ell=2,\ t_p=4\), the pilot clock equals the withheld-window clock. Each generator call takes one scalar \(t_p\). The schema can carry a different timestamp on each reference, but the seven C4 arms never exercise this case. The released C4 reader accepts only an ID, a channel Boolean and a matched sum. Its fitted encoder emissions and quadrature are tied to \(t_p=4,\ \omega=+0.35,\ \sigma_\phi=0.10,\ \mathrm{SNR}=2\,\mathrm{dB}\). [Generator and arms](../../stage3d_qwen_signed_phase/data_stage3d.py) · [Schema](../../stage3_ministral_cache/schema.py) · [Reader](../../stage3c/A/q2_reader.py) · [Stage 3C derivation](../../stage3c/A/Q2.md)
research/stage4_math_and_design/M3/THEOREM_M3.md:7:A copy has the same acquisition ID and samples. A fresh acquisition has a new ID and independent noise conditional on the shared path. The schema checks duplicate channel flags, timestamps within \(10^{-12}\), and exact sample equality. The weighted pool also checks amplitude and noise variance, which affect the likelihood. Stage 3D's distractor has a separate random quadrant, phase, and noise. Its density cancels when calculating the target posterior. These results condition on public, exogenous acquisition design and correct channel declarations.
research/stage4_math_and_design/M3/THEOREM_M3.md:21:Here \(\sigma_m^2=\mathbb E|\eta_m|^2\) is complex matched-noise variance. The orthogonal sample coordinates are ancillary under the white circular model. Known nonwhite noise calls for the corresponding whitened matched inner product.
research/stage4_math_and_design/M3/THEOREM_M3.md:33:For fixed nonempty design, the ratio of densities of two datasets is constant in physical phase precisely when \(\Re[(S-S')e^{-i(bg+\theta)}]\) is constant. Comparing antipodal phases at both quarter-turn axes forces \(S=S'\), even when only four phase values are available. The likelihood-ratio criterion proves that **one complex number is minimal sufficient for the physical pilot likelihood**. Factoring and integrating proves sufficiency for \((X_1,X_3)\) and for the joint transmitted-label and withheld-sector target.
research/stage4_math_and_design/M3/THEOREM_M3.md:37:For S17 at the pinned initial phase, the circulant mapping is invertible in the underlying Gaussian model. In the first-window turn-zero region, \(\Re T>|\Im T|\) almost surely. On any code cell of positive probability within that region, the density for \(U=0\) exceeds the density for \(U=2\) pointwise because \(\Re T>0\). Also, the sum of densities for \(U=0,2\) exceeds the sum for \(U=1,3\) by the strict increase of \(\cosh\) in \(|\Re T|\) versus \(|\Im T|\). Integrate over the code cell to obtain the nonzero first and second harmonics of \(a_{z_1}\); the third is the conjugate of the first. The other turns just permute the same reasoning. Thus the normalized carrier posterior is itself a minimal Q2 statistic. The argument identifies that statistic even in the Brownian case, without using a global injectivity assumption for \(S\mapsto (L_g/\sum L_g)_g\). The first phase moment is nonzero for at least one \(z_2\) since its unconditional value has magnitude \(e^{-\sigma_\phi^2p/2}\).
research/stage4_math_and_design/M3/THEOREM_M3.md:56:At \(p=0\) or zero diffusion use the deterministic pilot phase. At \(p=h\) the conditional second-window phase is a point mass. Define \(d\eta_{z_2}(s)\propto dP(\Theta_p=s)\,\mathbb E[m_{z_2}(\Theta_h)\mid\Theta_p=s]\). Then
research/stage4_math_and_design/M3/THEOREM_M3.md:73:The audited generator is Brownian motion with drift. An OU model would use its own Gaussian mean, covariance and bridges. At sorted distinct pilot clocks \(t_j\), define the precision-weighted family
research/stage4_math_and_design/M3/THEOREM_M3.md:81:The family \((t_j,D_j)_j\) is sufficient given design. The extra \(P_j\) gives the complex precision at that clock and is useful for estimating the observation model or defining a coarsened likelihood. At known parameters it already influences the full posterior through the weights in \(D_j\).
research/stage4_math_and_design/M3/THEOREM_M3.md:91:The precision is tridiagonal. Its diagonal is \((d_j^{-1}+\mathbf1_{j<K}d_{j+1}^{-1})/\sigma_\phi^2\) and its adjacent off-diagonal is \(-1/(\sigma_\phi^2d_{j+1})\). A pilot at zero uses a deterministic node. Set \(c_h=(\sigma_\phi^2\min(h,t_j))_j\). The conditional \(B_h\mid v\) has mean \(c_h^\top C^{-1}v\) and variance \(\sigma_\phi^2h-c_h^\top C^{-1}c_h\). Between two adjacent clocks this is linear interpolation with variance \(\sigma_\phi^2(h-t_j)(t_{j+1}-h)/(t_{j+1}-t_j)\).
research/stage4_math_and_design/M3/THEOREM_M3.md:93:The exact conditional likelihood ratio for carrier labels \(g,g'\) is \(L_g/L_{g'}\), where
research/stage4_math_and_design/M3/THEOREM_M3.md:103:**Minimality for the physical likelihood.** A parameter-independent density ratio implies \(\sum_j\Re[(D_j-D'_j)e^{-i(bg+v_j)}]\) is constant. The phase vector has full support, so vary each \(v_j\) independently to force \(D_j=D'_j\). A deterministic zero-time node is resolved by comparing the four \(g\) values. At zero diffusion all residual phases coincide and the derotated sum is sufficient.
research/stage4_math_and_design/M3/THEOREM_M3.md:109:For at least one \(z_2\) of positive probability the bracket is nonzero, since averaging it over \(z_2\) yields \(e^{-\sigma_\phi^2t_1/2}-e^{-\sigma_\phi^2t_2/2}\ne0\). An informative \(z_1\) has \(A_1\ne0\). Continuity and positive Gaussian observation density extend the difference to positive conditional measure along equal-sum fibers for an open set of totals. Thus even the optimal posterior based on the sum has strictly worse expected log loss. The same proof works at \(\omega=0\). A joint target containing \(X_1\) inherits the loss through its \(X_1\) marginal.
research/stage4_math_and_design/M3/THEOREM_M3.md:113:The coherent approximation freezes residual pilot phases at the earliest clock and retains the correct \(Z_2\) model given that clock. For two clocks, conditional on \(v_1\), the increment has variance \(\xi=\sigma_\phi^2(t_2-t_1)\). For \(q=D_2e^{-i(bg+v_1)}\) and \(f(\delta)=\exp(2\Re(qe^{-i\delta}))\), suppressing the \(Z_2\) factor gives
research/stage4_math_and_design/M3/THEOREM_M3.md:118:The first correction already depends on the late pilot separately. For the joint \(Z_2\) likelihood compare the Gaussian covariances of \((B_h,B_{t_1},\ldots,B_{t_K})\) before and after freezing. Each entry changes by at most \(\sigma_\phi^2(t_K-t_1)\). Differentiating Gaussian expectations along the convex covariance interpolation gives half the covariance change contracted with the expected Hessian of the integrand, by Gaussian integration by parts. A positive diagonal regularization and continuity handle singular covariances. At bounded pilot sums, the integrand and its derivatives through second order are bounded. Consequently the integrated likelihood changes by \(O(\sigma_\phi^2\Delta t)\) at fixed \(K\). Log-likelihood ratios and normalized posteriors obey the same local order on compact sets with denominators bounded away from zero. This is an error bound for coherent collapse on bounded evidence. The expected NLL gap depends on signal strength, discrimination, and model mismatch.
research/stage4_math_and_design/M3/THEOREM_M3.md:120:There is also an optimal **coarsened-data** Bayes comparator. Conditional on \(v,g\), \(D_{\rm sum}\) is complex Gaussian with mean \(i^g\sum_jP_je^{iv_j}\) and variance \(\sum_jP_j\). Phase marginalization retains the squared modulus of that conditional mean in the exponent. A coherent single-clock model adds a likelihood approximation on top of discarding the individual clock sums.
research/stage4_math_and_design/M3/THEOREM_M3.md:167:**Blackwell.** Dropping the new acquisition maps the augmented experiment to the old experiment. There is an explicit garbling even after same-clock pooling. Given the augmented pooled sum \(S_+\) with total precision \(P_+\), the old sum of precision \(P_o\) has conditional law \(\mathcal{CN}((P_o/P_+)S_+,\,P_o(1-P_o/P_+))\), independent of the physical phase. This follows from joint Gaussian receiver noise with covariance \(P_o\) between the two sums. Across clocks, use this garbling within each clock and drop new-only clocks. An exact posterior preserves all decisions about its target, so the Bayes risk of the augmented posterior experiment improves weakly for every loss. The expected log-loss gain is \(I(X_1;T_{\mathrm{new}}\mid Z_1,Z_2,\mathcal R_{\mathrm{old}})\). For the finite-time Brownian Q2 model with informative \(a_{z_1}\) and a fresh finite positive-SNR pilot this is strict. At zero old evidence, the new pilot derivative of the first target moment is nonzero for some \(z_2\), since the unconditional \(e^{iB_{\mathrm{new}}}\) moment is positive. Continuity and positive observation densities give a positive-measure change in the posterior. This applies to the first pilot and subsequent ones at common or distinct clocks. A copy, independent distractor or zero-SNR acquisition gives equality. [Blackwell, *Equivalent Comparisons of Experiments*](https://doi.org/10.1214/aoms/1177729032)
research/stage4_math_and_design/M3/THEOREM_M3.md:169:For Q2 with these same nondegeneracy conditions, strict population MAP improvement also follows. At zero old evidence, choose a new pilot value giving a nonuniform posterior. Quarter-turning **only that new pilot** produces the four label rotations of that posterior, because the old likelihood is constant in \(G\) there. The intersection of the four sets of maximizers is empty. The strict gaps excluding each label from at least one such maximizer set persist in neighborhoods of the new values and of zero old evidence. The conditional new pilot has positive density in every neighborhood. Therefore the expected new maximum posterior probability strictly exceeds the maximum of the conditional mean on a positive-measure set of old observations. This strengthens the sample-level claim to a population statement for the specified model. Accuracy can still go down on individual mothers or a finite test set, and the result supplies no guarantee for a learned host. For Q3 alone, strict log-loss improvement requires its conditional sector distribution to change.
research/stage4_math_and_design/M3/THEOREM_M3.md:171:**Fisher information.** A discrete quadrant or symbol has no ordinary differentiable-parameter Fisher information. For continuous deterministic pilot phase \(\theta\), the score \(2\Re[(T-\mu)^*\partial_\theta\mu]/\sigma_m^2\) has variance \(2\alpha_m^2/\sigma_m^2\). The conditional phase information at clock \(j\) is exactly \(2P_j\), with strict increments for positive precision. For a continuous global phase parameter in a marginalized Brownian model the old score is the conditional expectation of the augmented score. The gain in Fisher information is the expected conditional variance, which is nonnegative and strict exactly when that variance is positive.
research/stage4_math_and_design/M3/reply_r1.md:17:**One complex precision-weighted sum per physical clock.** The sum is minimal sufficient for the pilot likelihood at one clock, and two real coordinates are locally necessary for a regular C4 Bayesian readout. The fixed-prior minimal statistic is the target posterior. At different clocks, the single derotated sum is strictly insufficient for Q2 under positive Brownian diffusion. The report derives the Wiener likelihood ratio, a local \(O(\sigma_\phi^2\Delta t)\) coherent-collapse bound, and the joint Q2/Q3_after_pilot readout. Under the stated Q2 model, a fresh valid pilot strictly improves population Bayes log loss and MAP accuracy. The conditional continuous-phase Fisher information increases by twice the new pilot's matched precision. Those guarantees apply to the model-based decision.
research/stage4_math_and_design/M3/reply_r1.md:36:- **M4**. Can Pluto/CTTC expose stable acquisition IDs, calibrated timestamps and matched-I/Q noise variance, plus CFO and phase-diffusion estimates or uncertainty? Confirm which pilots are available before the target deadline and how channel identity and cross-pilot noise dependence are established.
```

### Event #7: `item.completed` — `command_execution` (`item_6`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n '"'^def |''^class |''^#'"' research/stage4_math_and_design/M4/verify_m4.py research/stage4_math_and_design/M4/THEOREM_M4.md; rg -n 'matched|only.*same' 'C:/Users/李则徐/.codex/memories/rollout_summaries/2026-11-17T20-54-26-dFGP-p4_em_native_transformer_assessment_and_github_publication.md'"
```

**Aggregated Output:**

```text
research/stage4_math_and_design/M4/verify_m4.py:37:def branch_text(ref, folder, filename):
research/stage4_math_and_design/M4/verify_m4.py:43:def literal_constants(source, names):
research/stage4_math_and_design/M4/verify_m4.py:72:def noise(rng, shape):
research/stage4_math_and_design/M4/verify_m4.py:76:def gaussian_kl(mu, cov, mu0, cov0):
research/stage4_math_and_design/M4/verify_m4.py:83:def centered_t_kl(nu, m, trace_ratio, log_det_ratio):
research/stage4_math_and_design/M4/verify_m4.py:91:def assert_mc(values, expected, sd_multiple=6):
research/stage4_math_and_design/M4/verify_m4.py:98:def verify_distributions(rng):
research/stage4_math_and_design/M4/verify_m4.py:144:def posterior_and_mi(weights, labels):
research/stage4_math_and_design/M4/verify_m4.py:150:def channel_information(labels, codes, code_count):
research/stage4_math_and_design/M4/verify_m4.py:167:def encoder_metrics(encoder, windows, labels):
research/stage4_math_and_design/M4/verify_m4.py:184:def lift(x, ancillary):
research/stage4_math_and_design/M4/verify_m4.py:188:def frozen_encoder_sweep(rng, encoder):
research/stage4_math_and_design/M4/verify_m4.py:267:def nuisance_curves(rng, encoder):
research/stage4_math_and_design/M4/verify_m4.py:353:def guarded_intervals(ranges, circumference=None):
research/stage4_math_and_design/M4/verify_m4.py:362:def cubic_samples(center):
research/stage4_math_and_design/M4/verify_m4.py:366:def symbol_support(k, sps, fractional, offsets, base=0):
research/stage4_math_and_design/M4/verify_m4.py:370:def audit_local_support(unit, holdout, allowed_nuisance, refs, targets, sps, offsets):
research/stage4_math_and_design/M4/verify_m4.py:384:def audit_splits(anchors):
research/stage4_math_and_design/M4/verify_m4.py:487:def verify_manifests(check_http):
research/stage4_math_and_design/M4/verify_m4.py:542:def verify_budget(splits, lock):
research/stage4_math_and_design/M4/verify_m4.py:591:def main():
research/stage4_math_and_design/M4/THEOREM_M4.md:1:# Track M4 · Real SDR bridge for Stage 4
research/stage4_math_and_design/M4/THEOREM_M4.md:7:## 1. Observation laws and losses
research/stage4_math_and_design/M4/THEOREM_M4.md:9:### 1.1 The three representations
research/stage4_math_and_design/M4/THEOREM_M4.md:29:### 1.2 Conditional KL and perturbation terms
research/stage4_math_and_design/M4/THEOREM_M4.md:74:### 1.3 Student scale, Rayleigh scale, and the frozen nonlinearity
research/stage4_math_and_design/M4/THEOREM_M4.md:112:## 2. Hardware benchmark and causality contract
research/stage4_math_and_design/M4/THEOREM_M4.md:114:### 2.1 Candidate rows and time order
research/stage4_math_and_design/M4/THEOREM_M4.md:129:### 2.2 What zero leakage requires and what was verified
research/stage4_math_and_design/M4/THEOREM_M4.md:145:### 2.3 Same-row ablations and pooling
research/stage4_math_and_design/M4/THEOREM_M4.md:158:## 3. Frozen compute envelope and dispatch
research/stage4_math_and_design/M4/THEOREM_M4.md:160:### 3.1 Dataset inputs and handoffs
research/stage4_math_and_design/M4/THEOREM_M4.md:172:### 3.2 Model and scored workload
research/stage4_math_and_design/M4/THEOREM_M4.md:180:### 3.3 Budget arithmetic and zero-bill gate
research/stage4_math_and_design/M4/THEOREM_M4.md:202:## 4. CPU replay and interpretation
22:- P4-B original benchmark: 23 condition/SNR cells, 7,176 coded blocks per receiver. Original EMFloatFormer2D errored on 4,476 blocks; fixed-prior 2D-LMMSE on 4,307; matched-statistics LMMSE on 2,548.
24:- Follow-up receiver experiments reduced errors on unseen delay distributions, but the pure Fourier/逐径 pilot front-end alone achieved nearly all of the gain. Therefore do not attribute this improvement to Transformer attention without matched-front-end controls.
36:- Formal two-dimensional result: EM did not significantly outperform fixed-prior classical baselines in any primary comparison. In-distribution at 10 dB/strong PA, EM had 3,266/4,160 block errors versus fixed LMMSE 3,227 and fixed IDD-PA 3,211; EM was significantly worse than fixed IDD-PA after Holm correction (`p=0.0234375`). Long-delay EM decoded only 22/4,160 blocks successfully, though matched-statistics baselines were much better.
65:Conclusion: P4 made substantial algorithmic progress by implementing and testing a complex-valued, gauge-equivariant, Hermitian-attention Transformer prototype with physical channel simulation and coded-link evaluation. The strongest evidence is that signal symmetries and complex-valued operations were encoded directly into the model and some mechanisms showed reproducible value in controlled 1D ablations. However, the formal 2D result did not establish a learned-Transformer advantage; the largest 2D improvements came from better pilot/channel front-ends. The next decisive experiment is a strictly matched comparison where EM and classical receivers share the same physical front-end, pilot information, decoder, and inference budget, isolating only the learned module.
```

### Event #8: `item.completed` — `command_execution` (`item_4`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "Get-Content -Encoding utf8 research/stage4_math_and_design/M5/reply_r1.md; rg -n 'angle|log_radial|quality|quantiz|pack|legacy|[-+]\\s*3' research/stage4_math_and_design/M5/verify_m5.py"
```

**Aggregated Output:**

```text
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
40:    ("flat_minus3", [-3] * 4, [1] * 4, [-3, 0, 2, -1], [1] * 4, -3),
43:    ("selective_fade", [14, -3, 3, 24], [1, .23, .42, 1.20], [7, 1, 7, 3], [1, .4, .85, .6], 0),
118:def quality_edges(bits):
131:    local_angle = (np.angle(t * (1j)**(-turns)) + np.pi / 4) / (np.pi / 2)
132:    packets = {"4_s17": original}
134:    for method, (angle_bits, radial_bits, quality_bits) in LAYOUTS.items():
135:        angle = np.clip(np.floor(local_angle * 2**angle_bits).astype(int), 0, 2**angle_bits - 1)
137:        quality = np.searchsorted(quality_edges(quality_bits)[1:-1], residual / variance)
138:        bins[method] = (turns * 2**angle_bits + angle, radius, quality)
139:        extra = radial_bits + quality_bits
140:        tail = (angle << extra) | (radius << quality_bits) | quality
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
166:def polar_emissions(angle_bits, radial_bits, mean_amp, noise_ratio, order=18):
167:    """Integrate the complex Gaussian probability of each packed polar cell."""
172:    angles = 4 * 2**angle_bits
173:    width = 2 * np.pi / angles
174:    symmetry = gcd(PHASES, angles)
176:    angle_shift = angles // symmetry
177:    result = np.zeros((base_phases, angles, len(r_edge) - 1))
178:    phase = np.angle(UNIT[:base_phases])[:, None, None]
183:    angular_node = -np.pi / 4 + (np.arange(angles)[:, None] + shifts) * width
195:    return np.concatenate([np.roll(base, shift * angle_shift, axis=1) for shift in range(symmetry)])
198:def window_log_likelihood(method, t, residual, packets, bins, a, v, encoder, key):
209:                    -(distance + residual[:, window, None]) / variance - 3 * np.log(variance)
217:                values[:, window, noise_id] = np.log(emission[:, packets[method][:, window]].T)
219:                angle_bits, radial_bits, quality_bits = LAYOUTS[method]
220:                polar = polar_emissions(angle_bits, radial_bits, a[window] / np.sqrt(variance), multiplier)
221:                gamma_mass = np.diff(gammainc(2, quality_edges(quality_bits) / multiplier))
222:                angle, radius, quality = (value[:, window] for value in bins[method])
223:                values[:, window, noise_id] = polar[:, angle, radius].T + np.log(gamma_mass[quality, None])
234:        angles = (old_phase[:, window, None] + np.arange(4) * QUARTER) % PHASES
235:        ll = windows[:, window, noise[:, window, None], angles]
455:    packets, bins = encode(all_iq, t, residual, v, encoder)
471:                raw = raw / var - 3 * np.log(var)
474:        ll = window_log_likelihood(method, t, residual, packets, bins, a, v, encoder, key)
553:def legacy_inner_witness(encoder):
572:    raise AssertionError("Expected the legacy inner index to use more than matched sum and residual energy")
582:    inner_witness = legacy_inner_witness(encoder)
614:            "cache_extensions_angle_radius_quality_bits": LAYOUTS,
615:            "compact_reader": "integrated physical cells; packed legacy inner bits available but ignored",
```

### Event #9: `item.completed` — `command_execution` (`item_5`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "Get-Content -Encoding utf8 research/stage4_math_and_design/M6/reply_r1.md; rg -n '173|960|0.49|0.18|Tier|futility|10 seed|cluster|holdout|172|0.746' research/stage4_math_and_design/M2 research/stage4_math_and_design/M6 -g '*.md'"
```

**Aggregated Output:**

```text
# M6 Round 1 • Decision for the Stage 4 launch

**Use the direct cache readout as an absolute acceptance gate.** Stage 3D's phase arm beat its zero control on G1, yet both host arms fell far short of the direct cache readout. The verified G1 ladder is **64.641% / 0.929038 nats** (Full IQ) → **64.617% / 0.941126** (s17) → **23.799% / 1.445229** (Qwen phase). The measured increments are **0.012087838** (s17 vs Full IQ), **0.478025740** (Qwen at unit temperature vs s17), and **0.026076989** (locked temperature vs unit temperature). The Qwen phase arm's calibrated excess over s17 is **0.504102729 nats**, paired block 95% CI **[0.491294, 0.516912]**. These are observed score differences. The archived scores leave the separate connector and host conditional mutual informations unidentified.

| Q3 cell | s17 − Full IQ (nats) | Phase − s17 (nats) |
|---|---:|---:|
| G1 | .012088 | .504103 |
| G2 | .002884 | .313596 |
| G3 | .025652 | .461480 |
| G4 | .012113 | .318816 |
| H_c4_035 | .007505 | .466854 |

**Q2 C4 gives the decision-changing intervention.** A fresh reference took Work A from **66.406% / .819207** to **74.219% / .663189**. Qwen phase went from **28.776% / 1.388883** to **26.432% / 1.405640**. Deduplication gave exact output parity with `valid`. The raw duplicate had mean TV **.040377** from `valid` and reduced phase accuracy to **26.302%**. G1's worst phase co-rotation defect was **.380019** versus **3.33×10⁻¹⁶** for s17. The report contains both host arms and all seven Q2 controls. [Theorem and evidence](THEOREM_M6.md), [Full replay](results_m6.json)

## Release gates

- **Tier 0, CPU.** Freeze causal inputs and the physical group action. Compute Bayes-risk brackets and verify the cache can support the objective. Require exact group transport, ID deduplication, masked-distractor and padding invariance. A proven collision or infeasible risk bound ends that branch. A poor probe score leaves the Bayes ceiling open.
- **Tier 1, CPU.** Require a joint cache×phase contrast on held-out CFO and angle. Check the actual host's centered candidate-score Jacobian and its target subspace before buying adaptation. The toy probe gave **.935161** additive OOD RMSE and **7.38×10⁻¹⁶** with analytic coupling. It took under one second and establishes the obstruction for the additive score family. The actual pinned Qwen host still needs its own Jacobian and precision checks.
- **Tier 2, capped GPU.** Run one seed for **173/960 steps** on disjoint dev blocks. Test throughput, scorer parity, practical effect and predictive futility. A go buys continuation to three seeds and independent blind scoring. Freeze the real frontends on CPU. Score the same three frozen connectors on Pluto and CTTC only after parity and throughput fit the resource ledger.

**Compute decision.** In the 10,000-candidate mixture, sequential cost averaged **\$6.101243 vs \$17.12** for fixed, a **64.3619%** simulated reduction (**64.3232%** by Gaussian integration). Working power was **93.280%** (simulation) and **93.395%** (model). Least-favorable Type-I error before CPU screening was **4.4695%** with modeled binding futility and **5.0000%** with nonbinding futility. In the simulated null mixture, false efficacy was **0.800% vs 1.760%** for fixed. These figures depend on the modeled candidate mixture, screen sensitivity and pilot/final relationship. The Stage 4 cap is **\$9.50** with **\$3.38** left against the historical \$12.88. It covers synthetic G1–G4 and Q2_C4, 2,240 Pluto frames and 1,440 CTTC packets if measured resource use fits. Round 1 incurred **\$0.00** of cloud spend.

## Questions for M1–M5, Round 2

- **M1** • Which exact $C_4^2$ action maps the full input, including references, to answer permutations? Can a single transported host call pass exhaustive orbit and OOD rectangle tests, including stabilizers?
- **M2** • What independent cluster-level statistic and development variance model will justify launch power after a one-seed pilot? Which absolute cache-relative margins and multiplicity rules will govern release?
- **M3** • What minimal RefPool statistic preserves fresh-reference gain under correlated pilots, and what counterexample rules out mean or max pooling? Can IDs, masks and fixed-position prompts be verified byte-for-byte?
- **M4** • Which causal Pluto/CTTC folds can deliver a useful cache-risk bracket, and what same-workload timing and parity witness makes the L4/A10G plan viable within the row caps?
- **M5** • Which same-cache collision prevents a Stage 5 jump decision, and what is the cheapest extra read resolving it? Can you freeze one Stage 6 query common to 4B, 9B, 27B and LEMMO under matched evidence access?

Local replay passed on Python 3.12.7 with NumPy 2.3.5 and SciPy 1.17.0. The maximum `summary_master.json` accuracy/NLL discrepancy was **3.56×10⁻¹⁶** across 102 comparisons. [Verifier](verify_m6.py)
research/stage4_math_and_design/M6\THEOREM_M6.md:48:| H_c4_035 | 1,024 | 63.867% / .942628 | 63.770% / .950133 | 25.765% / 1.416987 | 23.172% / 1.435077 | .007505 | .466854 | .484944 |
research/stage4_math_and_design/M6\THEOREM_M6.md:50:For G1 the paired 64-mother block 95% intervals are **[.009178, .014998]** for s17 minus Full IQ and **[.491294, .516912]** for phase minus s17. Phase's observable telescope is .012087838 + .478025740 + .026076989 = .516190567 nats, using unit-temperature host scores for the middle contrast. About **97.66%** of phase's excess over Full IQ lies after the s17 readout when measured at locked temperature. The G1 phase-versus-zero uplift did meet Stage 3D's relative target, even though both arms exceeded the log-loss of a uniform predictor (1.386294). G1's co-rotation worst coordinate defect was **.380019** in phase and near machine precision for the direct readout. Offline 16-way group averaging brought phase NLL to **1.411312** and accuracy to **17.050%**. On the archive's sampled rotations these are distinct endpoints. The all-rotation `H_c4_035` row above is a fresh replay from the archive. Its identity-rotation, both-sign host metrics match Work F. The separate positive-sign Stage 3B lookup anchor in the master JSON is a different comparison.
research/stage4_math_and_design/M6\THEOREM_M6.md:66:## 2. Tier 0 • analytic rejection before any host work
research/stage4_math_and_design/M6\THEOREM_M6.md:75:- On finite locked support, form a simultaneous confidence set $\mathcal P$ for the true joint law of $(A,S)$, then use $\inf_{p\in\mathcal P}H_p(A\mid S)$ and $\sup_{p\in\mathcal P}H_p(A\mid S)$. For i.i.d. units, a Bonferroni union of exact binomial intervals for fixed atom probabilities, intersected with the simplex, is one valid construction. For independent clusters, use bounded cluster contributions to each atom and a simultaneous concentration bound. If the support is large the interval may be vacuous. Binning a continuous reference gives an entropy **upper** bound on the richer interface. A lower bound for the binned interface applies when the bins are what the host receives.
research/stage4_math_and_design/M6\THEOREM_M6.md:88:## 3. Tier 1 • CPU geometry and the frozen host
research/stage4_math_and_design/M6\THEOREM_M6.md:116:## 4. Tier 2 • sequential decisions and an exact cost envelope
research/stage4_math_and_design/M6\THEOREM_M6.md:126:This is optimal for specified priors, utilities, transition likelihoods and caps. Greedy information per dollar is at most a heuristic when checks are complementary. In a differentiable separable allocation with concave information utilities $v_j(n_j)$, constant per-unit prices $c_j$ and a binding budget, KKT gives $v'_j(n_j)/c_j=\lambda$ for interior arms, with only arms above this price at the origin receiving effort. For $v_j=-w_j\sigma_j^2/n_j$, the allocation is $n_j\propto\sqrt{w_j\sigma_j^2/c_j}$, with independent clusters and a fixed variance estimand. Use mandatory cohort minima and integer reallocation, especially for nine CTTC sessions. No universal zero-waste guarantee follows without a specified law. In a nonidentifiable case the valuable output is the collision and the minimum extra observation, even when no GPU trial follows.
research/stage4_math_and_design/M6\THEOREM_M6.md:134:Under the planning prior $\mu\sim N(0,4)$, $\mu\mid Z_1=z\sim N(v\sqrt tz,v)$ with $v=(1/4+t)^{-1}$, from completing the square. The predictive distribution for $Z_2$ has mean $\sqrt tz[1+(1-t)v]$ and variance $1-t+(1-t)^2v$. Continue if predictive $P(Z_2\ge b_2\mid z)\ge .10$, equivalently $z\ge -0.270438$; early efficacy proceeds to the full rollout for independent validation. The exact least favorable Type-I rate is **.050000** with nonbinding futility and **.044695** with modeled binding futility and no CPU screen. These are probability calculations under the specified Gaussian law; the Monte Carlo has sampling error.
research/stage4_math_and_design/M6\THEOREM_M6.md:152:**Actual Stage 4 release.** Use the pilot primarily for engineering, variance, and predictive futility on disjoint development mothers, sessions or blocks. Freeze pilot/final coupling and calibrate on external development simulations before using the two-look efficacy boundaries. If that law cannot be justified, allow only futility at the first look and apply a single valid, independent final test on blind blocks after all three seeds and scoring settings are fixed. Use new synthetic mother namespaces, disjoint from the Stage 3D diagnostic archives, and hold out all physical records used for the claimed final test. Conditional on the pilot and model selection, the blind test must be valid for the declared null. Its unconditional false-promotion rate is at most its level by iterated expectation. Power after futility, particularly with seed-to-seed heterogeneity, still needs development calibration. Full release also requires passing physical margins and negative controls. Count mothers or recording clusters as independent units under the registered assumptions. Use randomization inference where the experimental randomization supports it, or a declared block model with sensitivity to correlation. For Pluto, 16 temporal blocks support inference within the capture when justified. For CTTC, report 72 nested blocks and session-level (9-unit) sensitivity. A sign-flip test is exact for justified randomization or a stated joint sign-symmetry null. Mean-zero shifts alone supply neither condition.
research/stage4_math_and_design/M6\THEOREM_M6.md:160:| Download/reference alignment, causal frontends, dataset audit and weight staging to Volume | CPU | 7,200 | .00006172 | .444384 |
research/stage4_math_and_design/M6\THEOREM_M6.md:162:| Synthetic pilot, 1 seed ≤173 of 960 steps (18.02%) | A100-80GB | 650 | .00075572 | .491218 |
research/stage4_math_and_design/M6\THEOREM_M6.md:167:| Base compute envelope | | | | **8.305172** |
research/stage4_math_and_design/M6\THEOREM_M6.md:200:| 4A causal manifest | Assert disjoint recording/source intervals plus filter delays, immutable label/reveal permissions, session holdout rules, and null shortcuts. Pluto 850 MHz uses 2,240 frames in the historical `[4800,7040)` split with 16 chronological blocks. Check DC/IQ offsets, 20-frame hardware cadence and repeated symbols using chronology-only and metadata-only controls. CTTC must resolve Gray/bit offset, sample rate and filter timing causally. For true session transfer use predeclared whole-session outer folds and fit only on allowed sessions. A fixed, zero-shot configuration can evaluate all nine. Within-session held-out packets alone identify a different estimand. Preserve warm-up/guard and cyclic source-interval overlap checks. Assign reference IDs to physical acquisitions, not presentation slots. [Pluto branch report](https://github.com/20171806011/EM-FloatFormer/blob/ade8333fb68126251a83e484b69b821b5d97bcea/research/realdata_pluto_qpsk/REPORT.md), [CTTC branch report](https://github.com/20171806011/EM-FloatFormer/blob/1887aee6de9649bc0dd8a4f05fe3c4542f3f6d77/research/realdata_cttc_qpsk/REPORT.md) |
research/stage4_math_and_design/M6\THEOREM_M6.md:209:| 6E ultimate blind | CPU seal independent physical recordings and unseen question combinations. Verify time/evidence alignment, cluster count and attainable precision, multi-turn reveal provenance, baselines and matched scoring. Run once under a locked ledger and keep unverifiable tasks as explicit abstentions. |
research/stage4_math_and_design/M2\THEOREM_M2.md:105:3. **Hierarchy and multiplicity.** Keep all repeats, rotations, acquisitions, and matched arms inside their mother and higher cluster. Report a frozen-seed conditional estimate and, if claiming generalization to retraining, a separate random-seed inference. Use equal independent cluster weights for the following exact balanced analysis. Pluto supplies 16 units if they are independent captures. For a CTTC new-session claim use nine independent session means, with all eight clusters per session contributing to that session. A conditional claim on the fixed sessions needs the corresponding estimand and defensible within-session cluster independence. For a permutation or label-randomization gate, lock the allowed exchangeability orbit and recompute the same statistic throughout. Repeated/rotated labels move jointly with their mother and never split correlated session units as though independent. A synthetic randomized design with independent label assignments can instead rerandomize the sanctioned generator operation inside the original design. When there is no justified exchangeability, collect a randomized negative control. State whether it tests a sharp random-label null or a mean-effect margin. A permutation p-value \((1+\#\{T_{\rm perm}\ge T_{\rm obs}\})/(R+1)\) with uniform randomizations and ties included is conservative under the sharp design. Use exact enumeration when feasible.
research/stage4_math_and_design/M2\THEOREM_M2.md:132:For real data under the same Gaussian model with \(J_0\) independent sessions of \(m\) nested clusters and separate session and cluster effects, the session means have
research/stage4_math_and_design/M2\THEOREM_M2.md:134:V=\sigma_{\rm session}^2/J_0+\sigma_{\rm cluster}^2/(J_0m)+\sigma_\nu^2/S+
research/stage4_math_and_design/M2\THEOREM_M2.md:137:Use the same formulas with \(B=J_0\), \(\sigma_u^2=\sigma_{\rm session}^2+\sigma_{\rm cluster}^2/m\), and \(\sigma_e^2\) replaced by \(\sigma_e^2/m\); an additional session-by-seed effect is another contribution to the collapsed interaction variance. Use planned balanced session-level estimation or a separately validated unequal-weight design. For a fixed-seed claim with independent bounded cluster means and arbitrary within-cluster dependence, a one-sided Hoeffding bound gives an alternative to Gaussian t inference. For mean of \(B\) equal-weight cluster means in \([l,h]\), reject when the empirical excess exceeds \((h-l)\sqrt{\log(1/\alpha)/(2B)}\). For fixed weights \(a_b\) summing to one, replace \(1/B\) by \(\sum_b a_b^2\). Unknown non-Gaussian random seed effects with only three seeds do not admit a generic exact t power calculation. Correlation modeling and null randomization require separate justifications.
research/stage4_math_and_design/M2\THEOREM_M2.md:172:| G3 | +0.03708294 | +0.03751254 | −0.00001311 | −0.00000091 | −0.00041559 | −0.00042960 |
research/stage4_math_and_design/M2\THEOREM_M2.md:173:| G4 | −0.02491655 | +0.00143469 | +0.00443245 | +0.00002490 | −0.03080859 | −0.02635124 |
research/stage4_math_and_design/M2\THEOREM_M2.md:189:| G2 | zero | 1.308067 / .023020 | 1.306828 / .017347 | 1.220, .916, .711 |
research/stage4_math_and_design/M2\THEOREM_M2.md:193:| G4 | zero | 1.309751 / .032534 | 1.307715 / .021834 | 1.535, .831, .762 |
research/stage4_math_and_design/M2\THEOREM_M2.md:214:On G1 phase the arithmetic pool makes 8.76% of decisions better and 15.51% worse, the exact weighted hit changes here. Geometric makes 8.28% better and 15.72% worse. G1 phase F0 scores 20.88% / 1.411163. On complete orbits G1 phase's arithmetic gain is 0.03386703, while the reverse-barycenter KL is 0.03430077. For zero the corresponding numbers are 0.04120584 and 0.03488442. Restricting to the three saved rotations adds 0.00004913 and 0.00055046 to the loss gain. Across the tested cells pooled maximum equivariance and idempotence residuals were at floating-point precision. G1 raw maximum probability discrepancies were 0.48167 and 0.41957 for phase and zero. Structural compliance of the offline tables coexists with poor decoding and poor ECE.
research/stage4_math_and_design/M2\THEOREM_M2.md:220:G1 seed-mean orientation gains are +0.00981980 and +0.00504776. Their linear dependence on the probabilities makes them equal to the corresponding gains on the seed-average table up to rounding. NLL is nonlinear. Seed-mean swapped penalties are +0.01836983 and +0.00737184. Evaluating the arithmetic seed-ensemble table yields +0.01615141 and +0.00795537. The physical reference has orientation gain 1 by definition. Phase per-seed gains are −0.00147, +0.02104, +0.00989. The zero-arm gains are +0.00184, +0.01295, +0.00035. An orientation gain is a weighted projection coefficient onto the reference odd direction. It does not measure a fraction of the Bayes information or the Stage 3D arm contrast. The direct readout is a separate physical predictor with access to the known condition.
research/stage4_math_and_design/M2\THEOREM_M2.md:222:On all the saved G1 rotations of the phase checkpoint, a text-only flip raises NLL by +0.00323146 and a slot-only flip by +0.01723982, holding the other channel matched. A joint flip raises it by +0.01857416. The difference in differences is −0.00189712. On rotation zero these are +0.00302035, +0.01679462, +0.01836983, and −0.00144514. The interventions show sensitivity to physically aligned slots in the phase checkpoint, alongside a text response, in the tested models. They compare flipped slots to matched slots within a checkpoint. Training with the phase arm and training with the zero arm is a distinct intervention. A shared-development retraining comparison with a locked recalibration and independent seeds is needed to attribute persistent NLL gains to richer slot use.
research/stage4_math_and_design/M2\THEOREM_M2.md:231:| valid | .819207 | 1.388883 / 1.410690 | 1.383282 / 1.384875 | +.020214 | +.021807 |
research/stage4_math_and_design/M2\THEOREM_M2.md:232:| duplicate dedup | .819207 | 1.388883 / 1.410690 | 1.383282 / 1.384875 | +.020214 | +.021807 |
research/stage4_math_and_design/M2\THEOREM_M2.md:233:| duplicate raw | .819207 | 1.394420 / 1.413341 | 1.384193 / 1.385808 | +.017307 | +.018922 |
research/stage4_math_and_design/M2\THEOREM_M2.md:235:| valid plus independent | .819207 | 1.408232 / 1.418281 | 1.386244 / 1.385545 | +.010748 | +.010049 |
research/stage4_math_and_design/M2\THEOREM_M2.md:254:G1-proxy sample sizes and powers follow. CTTC uses session means across eight clusters per session. Its ICC parameter is the assumed fraction of the proxy block variance shared at session level, with no additional session-by-seed effect in this sensitivity run. “Minimum” means the first integer yielding at least 95% marginal power under the specified test and effects.
research/stage4_math_and_design/M2\THEOREM_M2.md:260:| 72 independent clusters sensitivity | 72 clusters | 7 / 5 | .212 / .293 | 15 / 7 | .978 / .998 |
research/stage4_math_and_design/M6\reply_r1.md:3:**Use the direct cache readout as an absolute acceptance gate.** Stage 3D's phase arm beat its zero control on G1, yet both host arms fell far short of the direct cache readout. The verified G1 ladder is **64.641% / 0.929038 nats** (Full IQ) → **64.617% / 0.941126** (s17) → **23.799% / 1.445229** (Qwen phase). The measured increments are **0.012087838** (s17 vs Full IQ), **0.478025740** (Qwen at unit temperature vs s17), and **0.026076989** (locked temperature vs unit temperature). The Qwen phase arm's calibrated excess over s17 is **0.504102729 nats**, paired block 95% CI **[0.491294, 0.516912]**. These are observed score differences. The archived scores leave the separate connector and host conditional mutual informations unidentified.
research/stage4_math_and_design/M6\reply_r1.md:17:- **Tier 0, CPU.** Freeze causal inputs and the physical group action. Compute Bayes-risk brackets and verify the cache can support the objective. Require exact group transport, ID deduplication, masked-distractor and padding invariance. A proven collision or infeasible risk bound ends that branch. A poor probe score leaves the Bayes ceiling open.
research/stage4_math_and_design/M6\reply_r1.md:18:- **Tier 1, CPU.** Require a joint cache×phase contrast on held-out CFO and angle. Check the actual host's centered candidate-score Jacobian and its target subspace before buying adaptation. The toy probe gave **.935161** additive OOD RMSE and **7.38×10⁻¹⁶** with analytic coupling. It took under one second and establishes the obstruction for the additive score family. The actual pinned Qwen host still needs its own Jacobian and precision checks.
research/stage4_math_and_design/M6\reply_r1.md:19:- **Tier 2, capped GPU.** Run one seed for **173/960 steps** on disjoint dev blocks. Test throughput, scorer parity, practical effect and predictive futility. A go buys continuation to three seeds and independent blind scoring. Freeze the real frontends on CPU. Score the same three frozen connectors on Pluto and CTTC only after parity and throughput fit the resource ledger.
research/stage4_math_and_design/M6\reply_r1.md:21:**Compute decision.** In the 10,000-candidate mixture, sequential cost averaged **\$6.101243 vs \$17.12** for fixed, a **64.3619%** simulated reduction (**64.3232%** by Gaussian integration). Working power was **93.280%** (simulation) and **93.395%** (model). Least-favorable Type-I error before CPU screening was **4.4695%** with modeled binding futility and **5.0000%** with nonbinding futility. In the simulated null mixture, false efficacy was **0.800% vs 1.760%** for fixed. These figures depend on the modeled candidate mixture, screen sensitivity and pilot/final relationship. The Stage 4 cap is **\$9.50** with **\$3.38** left against the historical \$12.88. It covers synthetic G1–G4 and Q2_C4, 2,240 Pluto frames and 1,440 CTTC packets if measured resource use fits. Round 1 incurred **\$0.00** of cloud spend.
research/stage4_math_and_design/M6\reply_r1.md:26:- **M2** • What independent cluster-level statistic and development variance model will justify launch power after a one-seed pilot? Which absolute cache-relative margins and multiplicity rules will govern release?
research/stage4_math_and_design/M2\reply_r1.md:10:| G4 | −0.024917 | −0.026351 | 0.179398 |
research/stage4_math_and_design/M2\reply_r1.md:13:Both Q3 arms observe the same signed metadata and codes. Bayes sign MI is available to both. On G2 and G4 zero retains its advantage under optimal scalar scaling, with \(0.031042\) and \(0.030809\) lower projected reverse-conditional KL. Work F's seed-mean orientation gain is 0.009820 for phase and 0.005048 for zero. Zero retains signed text. On G1 phase, holding the other path matched, flipping text costs 0.003231 nats and flipping slots costs 0.017240, with slot effects \(−0.000641,+0.035214,+0.017147\) across seeds. The within-checkpoint slot response does not establish a stable gain across retrains. On Q2 valid, recalibration reduces the phase lead from 0.021807 to 0.001594. The fresh-resample lead reverses from +0.000757 to −0.002672. All seven Q2 controls are in the full report.
research/stage4_math_and_design/M2\reply_r1.md:21:- **M4** should retain 131 synthetic blocks pending full-gate pilots. With G1 variance and 3 pp/0.04 nats true excess above the margins, the historical pair needs 7/5 independent blocks for 95% marginal power conditional on the three fixed seeds. Across retrains, three seeds cap even known-variance power at 71.54%/85.94%, regardless of block count. Ten paired seeds give proxy minima of 15/7 independent blocks or captures, or 7/5 CTTC sessions of eight clusters at proxy session ICC 0.10. At the observed G1 excess, the fixed-seed capture proxy needs 10/17. Treat Pluto's 16 captures and CTTC's nine sessions as separate design populations, measure their variance on independent pilots, and power every gate endpoint before locking real-data enrollment.
```

### Event #10: `item.completed` — `command_execution` (`item_8`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "Get-Content -Encoding utf8 research/stage4_math_and_design/M5/reply_r1.md; rg -n 'canon_angle|radial|quality|pack|quantiz|def .*code' research/stage4_math_and_design/M5/verify_m5.py research/stage4_math_and_design/M5/THEOREM_M5.md"
```

**Aggregated Output:**

```text
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
research/stage4_math_and_design/M5/verify_m5.py:110:def radial_edges(bits):
research/stage4_math_and_design/M5/verify_m5.py:118:def quality_edges(bits):
research/stage4_math_and_design/M5/verify_m5.py:126:def encode(iq, t, residual, variance, encoder):
research/stage4_math_and_design/M5/verify_m5.py:132:    packets = {"4_s17": original}
research/stage4_math_and_design/M5/verify_m5.py:134:    for method, (angle_bits, radial_bits, quality_bits) in LAYOUTS.items():
research/stage4_math_and_design/M5/verify_m5.py:136:        radius = np.searchsorted(radial_edges(radial_bits)[1:-1], abs(t) / np.sqrt(variance))
research/stage4_math_and_design/M5/verify_m5.py:137:        quality = np.searchsorted(quality_edges(quality_bits)[1:-1], residual / variance)
research/stage4_math_and_design/M5/verify_m5.py:138:        bins[method] = (turns * 2**angle_bits + angle, radius, quality)
research/stage4_math_and_design/M5/verify_m5.py:139:        extra = radial_bits + quality_bits
research/stage4_math_and_design/M5/verify_m5.py:140:        tail = (angle << extra) | (radius << quality_bits) | quality
research/stage4_math_and_design/M5/verify_m5.py:141:        packets[method] = turns if method == "2" else (original << (angle_bits + extra)) | tail
research/stage4_math_and_design/M5/verify_m5.py:143:            np.testing.assert_array_equal(packets[method] >> (angle_bits + extra), original)
research/stage4_math_and_design/M5/verify_m5.py:144:        assert np.all((packets[method] >= 0) & (packets[method] < 2**int(method)))
research/stage4_math_and_design/M5/verify_m5.py:145:        prefix = packets[method] if method == "2" else packets[method] >> (angle_bits + extra + 2)
research/stage4_math_and_design/M5/verify_m5.py:146:        unpacked = (
research/stage4_math_and_design/M5/verify_m5.py:147:            (prefix << angle_bits) | ((packets[method] >> extra) & (2**angle_bits - 1)),
research/stage4_math_and_design/M5/verify_m5.py:148:            (packets[method] >> quality_bits) & (2**radial_bits - 1),
research/stage4_math_and_design/M5/verify_m5.py:149:            packets[method] & (2**quality_bits - 1),
research/stage4_math_and_design/M5/verify_m5.py:151:        np.testing.assert_array_equal(unpacked, bins[method])
research/stage4_math_and_design/M5/verify_m5.py:152:        bins[method] = unpacked
research/stage4_math_and_design/M5/verify_m5.py:153:    return packets, bins
research/stage4_math_and_design/M5/verify_m5.py:156:def s17_emissions(encoder, rng, amplitude, variance):
research/stage4_math_and_design/M5/verify_m5.py:166:def polar_emissions(angle_bits, radial_bits, mean_amp, noise_ratio, order=18):
research/stage4_math_and_design/M5/verify_m5.py:167:    """Integrate the complex Gaussian probability of each packed polar cell."""
research/stage4_math_and_design/M5/verify_m5.py:170:    r_edge = radial_edges(radial_bits) / np.sqrt(noise_ratio)
research/stage4_math_and_design/M5/verify_m5.py:179:    radial_node = (r_edge[:-1, None] + r_edge[1:, None] + (r_edge[1:, None] - r_edge[:-1, None]) * node) / 2
research/stage4_math_and_design/M5/verify_m5.py:180:    radial_weight = (r_edge[1:, None] - r_edge[:-1, None]) * weight * radial_node / 2
research/stage4_math_and_design/M5/verify_m5.py:185:    for k, (r, rw) in enumerate(zip(radial_node, radial_weight)):
research/stage4_math_and_design/M5/verify_m5.py:186:        # Subdivide broad radial or angular bins for concentrated, high-SNR profiles.
research/stage4_math_and_design/M5/verify_m5.py:198:def window_log_likelihood(method, t, residual, packets, bins, a, v, encoder, key):
research/stage4_math_and_design/M5/verify_m5.py:217:                values[:, window, noise_id] = np.log(emission[:, packets[method][:, window]].T)
research/stage4_math_and_design/M5/verify_m5.py:219:                angle_bits, radial_bits, quality_bits = LAYOUTS[method]
research/stage4_math_and_design/M5/verify_m5.py:220:                polar = polar_emissions(angle_bits, radial_bits, a[window] / np.sqrt(variance), multiplier)
research/stage4_math_and_design/M5/verify_m5.py:221:                gamma_mass = np.diff(gammainc(2, quality_edges(quality_bits) / multiplier))
research/stage4_math_and_design/M5/verify_m5.py:222:                angle, radius, quality = (value[:, window] for value in bins[method])
research/stage4_math_and_design/M5/verify_m5.py:223:                values[:, window, noise_id] = polar[:, angle, radius].T + np.log(gamma_mass[quality, None])
research/stage4_math_and_design/M5/verify_m5.py:434:def run_profile(key, profile, tracks, encoder):
research/stage4_math_and_design/M5/verify_m5.py:455:    packets, bins = encode(all_iq, t, residual, v, encoder)
research/stage4_math_and_design/M5/verify_m5.py:474:        ll = window_log_likelihood(method, t, residual, packets, bins, a, v, encoder, key)
research/stage4_math_and_design/M5/verify_m5.py:526:def collision_witness(encoder):
research/stage4_math_and_design/M5/verify_m5.py:553:def legacy_inner_witness(encoder):
research/stage4_math_and_design/M5/verify_m5.py:614:            "cache_extensions_angle_radius_quality_bits": LAYOUTS,
research/stage4_math_and_design/M5/verify_m5.py:615:            "compact_reader": "integrated physical cells; packed legacy inner bits available but ignored",
research/stage4_math_and_design/M5/verify_m5.py:620:        "s17_radial_collision_witness": witness,
research/stage4_math_and_design/M5/THEOREM_M5.md:21:The families range over all permitted addresses, reference contexts, and candidate sets. Take $\tau=\varnothing$ for hypotheses without a jump, and replace $H$ by $(H,\tau)$ in the last term when the design target includes localization. Under a common dominating reference measure with positive conditional densities, identify reference kernels up to common null sets. For singular or deterministic side information, use the supported contexts and corresponding equality of kernels. The equivalence $y\sim y'$ iff their required kernels agree defines the minimal statistic. $\mathcal K$ is a representation of that quotient.
research/stage4_math_and_design/M5/THEOREM_M5.md:84:The optimum depends on the full prior and conditional laws. Varying the hypothesis prior or making all rival emissions identical can change the frontier at the same $B,\rho_p,d_p$. Fading, old SNR, covariance and trajectory priors cannot be inferred from those three arguments. In the nondegenerate noisy example, even the single-task Q1 excess is strictly positive at any finite alphabet size $m$. Choose $m+1$ sufficiently separated support points of the continuously varying posterior, and positive-mass disjoint neighborhoods around them. With only $m$ reproduction distributions, one neighborhood must remain at a positive distance from every reproduction, yielding a positive uniform lower bound by Pinsker's inequality.
research/stage4_math_and_design/M5/THEOREM_M5.md:94:For balanced binary hypotheses and 0–1 loss, the exact discriminability loss is $\frac12[\mathrm{TV}(P_0^Y,P_1^Y)-\mathrm{TV}(P_0^Z,P_1^Z)]$. Bayes accuracy is $(1+\mathrm{TV})/2$, and replacing the IQ densities by their cell masses replaces $\int |p_0-p_1|$ by $\sum_C|\int_C(p_0-p_1)|$. The triangle inequality is strict exactly when a cache cell contains positive mass from both strict signs of the likelihood difference. Hence log risk can increase while accuracy stays equal if within-cell likelihood changes leave the winning hypothesis unchanged.
research/stage4_math_and_design/M5/THEOREM_M5.md:104:The released s17 training generator draws coherent unit-energy QPSK with **additive Gaussian noise** at 2 dB. It uses three complex samples with mean magnitude $1/\sqrt3$ and fits to analytic noisy-label posteriors. The b8 name means eight bits for the pair. The turn comes from the sum, with a first-nonzero-sample fallback if that sum is zero. The encoder rotates the three samples before the six-input neural network and four-center assignment. The learned inner label is a codebook index. An angular quantizer cannot in general be assumed to produce that index. Stage 3C shifts Q1 by the difference of turns and Q2/Q3-repeat by the first turn. Its Q2 kernel deduplicates acquisition IDs, ignores known independent references, and is calibrated for its pinned pilot time and SNR. The signed-CFO Stage 3D evaluates s17 outside the encoder's coherent training condition. [Encoder and training](../../matched_cache/learned.py) · [Training driver](../../matched_cache/train.py) · [Paired generator](../../fourphase_paired_questions/paired.py) · [Stage 3C quotient](../../stage3c/G/quotient.py) · [Stage 3C Q2 reader](../../stage3c/A/q2_reader.py) · [Stage 3D data](../../stage3d_qwen_signed_phase/data_stage3d.py)
research/stage4_math_and_design/M5/THEOREM_M5.md:108:For a decoded matched sum $\hat T$, a conservative bound makes the precision tradeoff explicit. Let $e_U$ be the error of the surrogate log likelihood up to an additive constant shared across states. Conditional posterior KL to the surrogate is $E_p e_U+\log E_p e^{-e_U}\le(\sup_U e_U-\inf_U e_U)^2/8$ by Hoeffding's lemma. Its expected value upper-bounds the Bayes task loss, by data processing and the optimality of the cell decoder. The Gaussian mean terms give $e_U=2\sum_k \Re[(T_k-\hat T_k)^*m_k(U)]/v_k$ at fixed variance. Coarse rounding can therefore be more costly at higher precision. If angle error is at most $h/2$ and radial error at most $\epsilon_r$ on a bounded unsaturated bin, $|T-\hat T|\le\epsilon_r+|T|h/2$. Saturation and covariance terms must be handled separately.
research/stage4_math_and_design/M5/THEOREM_M5.md:110:**What s17 permanently merges.** Opposite means have likelihood ratio $\exp(4\Re[T^*\alpha]/v)$. Two interior points of one s17 cell with distinct projection on $\alpha$ give different likelihood ratios. A strict neural assignment margin ensures an open neighborhood around each point inside the same cell. Gaussian positivity and strict propriety then imply positive Bayes loss when a query separates these posteriors, including an informative phase pilot under the conditions of §1.2. The experiment verifies such a collision with the pinned weights. The same argument holds when an event or forecast contrast varies within a cell. In the variance experiment a circular pure-noise window has a uniform 2-bit turn at every variance, whereas radial or residual energy gives positive class information. s17 can retain some quality information through its learned index, but cannot preserve a continuously varying likelihood ratio. Bounds for the actual Pluto and CTTC acquisitions require their measured calibration and held-out data. The $-3$ to $+24$ dB and $>10$ dB gain variations in this report are synthetic stress conditions.
research/stage4_math_and_design/M5/THEOREM_M5.md:114:Retain the original 2-bit turn and 2-bit learned inner index as a prefix. Add canonical phase, a calibrated log radial/reliability field, and residual quality. The Gaussian reference implementation uses $r=|T|/\sqrt{v_{\rm nom}}$, a unit canonical phasor with angle $\arg T-\pi\,\mathrm{turn}/2$ in $[-\pi/4,\pi/4)$, $\log(1+r)$ radial bins, and $E/v_{\rm nom}$ residual bins. The likelihood concentration is $2|\alpha T|/v$ if the gain and variance are known. The radial field alone does not determine both unknown gain and unknown variance. Carry calibrated estimates as metadata or additional bits, or infer them jointly using the cached evidence. Keep original observation units for amplitude forecasts.
research/stage4_math_and_design/M5/THEOREM_M5.md:116:| Payload bits/window | Turn | Legacy inner | Canonical angle | Log radial/reliability | Residual quality |
research/stage4_math_and_design/M5/THEOREM_M5.md:125:Use the same boundary convention as the historical encoder, define a missing/invalid state, and compute the prefix in the encoding pass. Right-shifting by the extension length recovers s17 exactly for every supported input. This also preserves the original reader and its edge cases. Lossy polar coordinates alone cannot guarantee recovery of a learned label computed from the full raw sample vector. If a real-data adapter lacks a valid historical triplet, mark the legacy interface unavailable. For a Bayes-optimal decoder, an extended packet can always emulate the legacy reader. A strict gain for a query requires a refinement that divides some old cell into distinct conditional laws of positive mass. The benchmark reader uses the physical fields and leaves the legacy inner index unused, while still counting its bits. Its results therefore quantify an achievable decoder rather than the optimal legacy-plus-physical reader. The 12- and 16-bit residual thresholds are not nested. The 24-bit layout is proposed and has no risk estimate in this round.
research/stage4_math_and_design/M5/THEOREM_M5.md:127:Version the header with time origin/sample rate, template, gain and noise convention, calibration, quantizer edges, format ID, and capture/sample-span mapping. Record whether quality indicates clipping, noise, or a typed nuisance projection such as DC/spur. When required include irregular addresses, masks and per-window calibration bits in the budget. The CPU benchmark holds the address map and nominal amplitude/noise profile common to all readers and counts only the cache payload.
research/stage4_math_and_design/M5/THEOREM_M5.md:133:| EquivariantEvidenceConnector | Decode complex evidence, time, precision, quality, and masks. Return physical likelihood factors or calibrated candidate probabilities. Transform the reference and trajectory factors consistently under shared or per-window coordinate changes. |
research/stage4_math_and_design/M5/THEOREM_M5.md:209:Run from the repository root with the pinned NumPy and SciPy environment. The sandbox run used copies of the pre-existing package cache inside M5 and the same pins.
research/stage4_math_and_design/M5/THEOREM_M5.md:221:- **M4.** Use session- or acquisition-disjoint tuples. Calibrate SNR, gain and quality from allowed old data or a separate calibration split. Stratify natural, injected, and independent acquisitions, including low SNR and high fading. Test typed DC/spur projections when a scalar residual fails. Use a separately labeled oracle-calibration arm where needed.
```

### Event #11: `item.completed` — `command_execution` (`item_9`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n 'PhaseContext|condition|variance|schema|RefSufficientPool|canonical|timestamp' research/stage4_math_and_design/discussions/M3/prompt_r2.md; Get-Content -Encoding utf8 research/stage4_math_and_design/M3/THEOREM_M3.md | Select-Object -Skip 183 -First 31"
```

**Aggregated Output:**

```text
5:## 1. Cross-Track Demands on `RefSufficientPool` (From M1, M2, M4, M5, M6)
8:   - Furthermore, real hardware pilots at nearby lags (e.g., CTTC `126, 127` adjacent symbols through `256`-tap RRC pulse / residual ISI / shared CFO error $\epsilon_\omega$) can share **estimation uncertainty or noise covariance** $K_{\text{ref}}$ AND may use known QPSK pilot roots $d_m \neq 1$!
9:   - **Crucial Warning from M4**: `RefSufficientPool` MUST NOT double-count observations already conditioned inside M4's $\mathcal{F}^0$ causal state! When adding `(R_1, R_2)` onto $\mathcal{F}^0$, `RefSufficientPool` must emit the **conditional likelihood / innovation factor** given the prior state (and explicitly track which `acquisition_id`s already contributed to $\mathcal{F}^0$!).
11:   - M1 asked: can `RefSufficientPool` accept per-pilot timestamps (including **$t_p > t_3$**!), known pilot symbols $d_m$, shared CFO uncertainty $(\hat\omega, v_\omega)$ (remember CFO rate error scales phase variance as $v_\omega (t_j - t_i)^2$, creating cross-clock quadratic coupling beyond Brownian diffusion $\sigma_\phi^2 |t_j - t_i|$!), AND return joint factors `canonical_q2[4]`, `canonical_q3[4]`, and optional `joint[4,4]` relative to receiver frame $a_1$ (while keeping exact zero-influence distractors and byte-identical fixed-position prompt tokens)?
13:   - Before revealing candidate window $a$, M5 computes $f_{ha}(x) = \int p(Y_a = x \mid u, h, z) dP(u \mid z, h)$ and $\text{EIG}(a \mid z) = I(H; Y_a \mid z)$, THEN reveals ONE sealed acquisition $(id_a, t_a, y_a)$ into `RefSufficientPool`! Notice if `RefSufficientPool` represents each candidate hypothesis / phase-node state as clean `(phase_grid, log_weights_by_turn[4, N_nodes], clock_meta)`, then BOTH M5's pre-reveal $\text{EIG}$ integral AND post-reveal Bayes update share ONE exact mathematical engine!
14:4. **From M2 & M6 (Idempotence, Byte-Exact Prefix Invariance & Calibrated Comparison)**:
22:   - Derive the exact innovation factorization $p(\mathcal{R} \setminus \mathcal{F}^0 \mid \mathcal{F}^0, G, \boldsymbol{\phi}, \omega)$ when causal history $\mathcal{F}^0$ provides prior $(\hat\omega_0, v_\omega, P_{\phi,0})$ and new pilots have covariance $K_{\text{ref}}$: prove the **exact Schur-complement innovation sufficient statistic** and prove strictly what happens if an observation already in $\mathcal{F}^0$ is accidentally re-counted (`variance underestimation / overconfident false calibration KL`!).
23:   - Derive the augmented Gaussian covariance $(B_{t_j} + \tilde\omega t_j)$ combining Brownian increments $\sigma_\phi^2 \min(t_j, t_k)$ AND shared linear CFO prior variance $v_\omega t_j t_k$ (note rank-1 rank addition cleanly solved by Sherman-Morrison or appending 1 Gaussian-Hermite quadrature node over $\tilde\omega$!).
24:2. **Freeze the Unified Multi-Stage `RefSufficientPool` Schema** serving **M1 (`EquivariantEvidenceConnector`) + M4 (Pluto `15,18`, CTTC `126,127`, duplicate/missing provenance) + M5 (Stage 6 pre-reveal mixture & EIG)** simultaneously!
```

### Event #12: `item.completed` — `command_execution` (`item_7`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "Get-Content -Encoding utf8 research/stage4_math_and_design/M1/THEOREM_M1.md | Select-Object -Skip 152 -First 37; rg -n 'PhaseContext|Arm|omega_hat|variance' research/stage4_math_and_design/discussions/M1/prompt_r2.md"
```

**Aggregated Output:**

```text
\mathbb E[-\log (Pf)_Y(X)]\le\mathbb E[-\log f_Y(X)].
\]

**Proof.** Reindex the group sum after replacing \(x\) by \(hx\), then factor out \(R_{\chi(h)}\). An already equivariant \(f\) is fixed. For the inequality, apply convexity of \(-\log\) to the average of \(f_{y+\chi(g)}(gx)\). Its upper bound is the mean raw transformed loss. Invariance of the test law gives the displayed result after integration. With a finite incomplete rotation set the empirical comparison can have either direction. This proof uses log loss on positive probabilities. \(\square\)

Top-1 accuracy can fall on a complete orbit. Use aligned target 0. At three quarters of the orientations, assign the correctly classified vector \((0.35,0.34,0.16,0.15)\), and at the other quarter assign \((0.01,0.97,0.01,0.01)\). The average is \((0.265,0.4975,0.1225,0.115)\). Accuracy drops from \(75\%\) to zero while NLL drops from \(1.93866\) to \(1.32803\). Work G reproduces these metric directions on its three sampled rotations: NLL \(1.44523\to1.41131\), and accuracy \(23.80\%\to17.05\%\). The paired 95% interval for \(P f\) minus raw accuracy is \([-7.09,-6.41]\) percentage points. The verifier replays the calculation from the released raw score tables and locked temperatures. Symmetry projection alone does not restore the signed physical decision. [Stage 3D projection and metrics](../../stage3d_qwen_signed_phase/recompute_all.py)

## 4. Stage 4 `EquivariantEvidenceConnector`

1. **Canonicalize the inputs.** Accept integer `codes[B,2]`, a task enum, `PhaseContext`, and optional M3 evidence. Compute \((u,v)\) and the task's output shift. The branch sees only canonical inputs, invariant context, and fixed-frame pilot evidence. Return exact uniform probabilities immediately for `q3_indep`, and for `q2` when there is no usable shared pilot.
2. **Couple evidence and phase.** Start with canonical source distributions, initializing from the frozen S17 table. Propagate suitable circular moments with \(U_n(d)\) on each edge. For instance, transport \(m_{1,n}\) forward to the repeat target and de-rotate the second-window evidence with the known signed mean, while marginalizing uncertainty with nonnegative diffusion. Feed the real and imaginary components, source probabilities, and physical head features into the **same** trainable token projector. For exact continuous likelihood work, integrate a positive density or use Fourier harmonics appropriate to the sector and pilot maps. A first-moment feature alone cannot preserve a multimodal likelihood.
3. **Emit packed soft tokens.** Use a fixed 128-wide evidence vector and a 16-wide task embedding. Map through `Linear(144,128)`, GELU, and `Linear(128,1024)`, then reshape to eight 128-wide tokens. Apply a shared `Linear(128,2560)` and `LayerNorm(2560)`. Use a fixed task prompt and canonical candidate order in the existing Qwen four-candidate scoring interface. Train through the input embeddings with host weights frozen. Equivalent items can share an evaluation when they have identical canonical evidence and context.
4. **Use a bounded physical readout.** For Q3 repeat, the tested CPU implementation interprets canonical host scores as \(\delta=0.15\tanh(\ell_{\rm host})\), then transports the source \(\operatorname{softmax}(\log A_u+\delta)\) through the AWGN/Brownian kernel. A host bypass returns the direct S17 baseline. Q1 reads a relative root difference, Q2 reads a root relative to the common pilot frame, and post-pilot repeat integrates over the target phase conditional on the M3 factors. For an exact transport law as \(\theta\) changes with upstream evidence fixed, keep the source reweighting independent of target transport, or use the bypass. The tested residual reads coupled phase features, so it can change the source adjustment as \(\theta\) changes. The discrete \(H\) law holds for the composite in either case.
5. **Normalize once and lift.** Gather canonical probabilities into the physical choice order. For bitwise table parity, evaluate each distinct representative once at fixed context and reuse its canonical vector. Independently evaluated FP32 rows may differ by roundoff due to batch shape or kernels. Apply the same construction for candidate logits when used. Unrestricted free-text generation is outside the guarantee.

The design budgets 486,740 trainable parameters, counting two \(4\times16\) code embeddings, the \(5\times16\) task embedding, the projector and normalization, and an optional \(128\to4\) head. This leaves room under 2.2M for M3 calibration. A physical-head-first schedule permits a host-free starting point, but frozen-Qwen convergence and Modal free-credit runtime need measurement. The CPU verifier uses a shared frozen MLP and B's physical S17 prior. The A/B result combines quotient sharing, coupled tokens, and the physical head.

### M3 interface — `RefSufficientPool`

- Per acquisition accept `acquisition_id`, `shares_channel`, `timestamp`, `snr_linear`, known pilot root or sequence, and `matched_T` in the declared receiver frame. Deduplicate by acquisition identity before combining likelihoods. Keep fresh independently acquired pilots, including those at a shared timestamp. Independent-channel rows have no shared-channel likelihood factor under the stated model.
- Return `valid_mask[B]`, `pilot_frame_turn[B]` (zero under the fixed external frame), and `log_likelihood(phase_nodes, channel_turns) -> [...,4]` or equivalent factors grouped by timestamp. Equal-time independent AWGN pilots with known roots \(d_i\) reduce to \(S_p=\sum_i \rho_i T_{p,i}\overline{d_i}\) and \(\log L_p(g,\phi)=2\Re[S_p i^{-g}e^{-i\phi}]+C\). Retain relative likelihood scales across phase nodes and channel turns. Differently timed pilots use a shared Brownian chain in chronological order, which handles \(t_p>t_3\) without assigning a winding number to a wrapped observation.
- Q2 uses \(S_1-G\) with output shift \(a-r_p\). Post-pilot repeat uses the conditional target phase with shift \(a\). M1 owns the choice ordering and final gather. An isolated marginal on \(G\) discards dependencies needed by other tasks, and a single mean phasor can discard the sector modes.

### M4 interface — `PhaseContext` for Pluto and CTTC

- Supply `t1`, `t2`, `t3` and per-pilot timestamps in a common time unit and clock basis. Supply signed `omega_rad_per_unit` under \(T(t)\propto e^{+i\omega(t-t_1)}\), `diffusion_var_per_unit`, `snr_linear` with matched-filter normalization, and the frame and tracking state. Synthetic mapping is `t1=0`, `t2=h`, `t3=h+ell` and variance rate \(\sigma_\phi^2\). CFO in Hz uses \(\omega=2\pi f_{\rm CFO}\) with time in seconds. If the front end already de-rotates I/Q, pass the residual under the same sign convention.
- Supply an optional posterior on shared CFO and its estimate age or covariance if available. For independent Gaussian rate uncertainty of variance \(v_\omega\), the single-edge attenuation includes \(e^{-n^2v_\omega d^2/2}\). The same uncertainty is correlated across edges. Keep a shared latent or a joint law for multi-window and pilot inference. A rate uncertainty uses \(d^2\), whereas Brownian phase variance uses \(d\).
- Expose observation confidence and erasures, and specify any mismatch to the three-sample encoder, matched-filter normalization, circular noise, or drift law. Code-space equivariance is independent of the hardware label accuracy.

## 5. CPU verification and Round 2 decisions

Run from the repository root.

~~~powershell
uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 --with torch python -B research/stage4_math_and_design/M1/verify_m1.py
~~~

6:1. **M2 (Stage 3D Forensics)**: On `blind_q3_scores.npz`, `G1` locked `NLL(zero) - NLL(phase) = +0.035650` splits into `+0.036543` scalar temperature mismatch and **`-0.001017` reverse-conditional KL**; after seedwise scalar recalibration, **`zero` beats `phase` across all 5 cells** (`G1: -0.000893`, `G2: -0.027716`, `G3: -0.000430`, `G4: -0.026351`, `H: -0.003695`), despite `0.3631 nats` signed CFO MI available in `s17`! M2's Stage 4 gate requires post-dev-calibration NLL/Acc/NIG ($\eta_{\min}=0.10$), ECE $\le 0.05$, equivariance defect $\le 10^{-5}$, and fixed-text slot interventions.
8:3. **M1 Self-Critique + M2/M6 Challenge**: In your Round 1 `B_EquivariantPhaseCoupled`, using the physical S17 prior + residual `delta = 0.15 * tanh(ell_host)` reached `64.62%` (parity with `direct_s17`), leaving unresolved whether frozen host attention learned anything beyond the physical prior! Also naive `A` reached `62.40%` on a tiny toy MLP, so we need a clean causal isolation between **quotient equivariance**, **multiplicative phase coupling**, **coupled tokens into frozen host (`NO direct logit bypass`)**, and **physical prior + bounded host residual**.
12:   - M4 requires: (a) clean separation between **receive-gauge rotation** ($r_{\text{rx}}$: `Q1` TX diff invariant, `Q2` TX label invariant, `Q3` RX sector rolls $+r_{\text{rx}}$) and **TX relabeling**, (b) CFO uncertainty variance $v_\omega$ (giving edge damping $e^{-n^2 v_\omega d^2 / 2}$ correlated across edges!), and (c) SNR-preserving R3 lift on real Pluto/CTTC frames.
16:1. **The 4-Arm Causal Ablation Architecture (`Host-Pure` vs `Physical+HostResidual`)**:
18:     - **`Arm 0` (Stage 3D Baseline)**: non-quotient separate `<W1><W2><PHASE>`, no bypass.
19:     - **`Arm 1` (`Equivariant-Host-Pure`)**: quotient canonicalization (`canonical_input` + `rotate`) + multiplicative phase-coupled continuous summary fed purely as **soft tokens into the frozen host**, with **NO physical logit bypass** (`q_host` computed strictly from frozen LM / frozen surrogate logits, with isotropic `W_out` geometry)!
20:     - **`Arm 2` (`Equivariant-Physical-Residual`)**: canonical physical prior + bounded trainable host residual (`delta = gamma * tanh(ell_host)`), plus fixed-text counterfactual slot-swap and shuffled-slot diagnostics isolating host contribution beyond the analytic bypass!
21:   - Implement and run **BOTH `Arm 1 (Host-Pure)` AND `Arm 2 (Physical+Residual)`** in `verify_m1.py` on the SAME frozen CPU surrogate host and identical `|\omega| <= 0.25` training split, testing `G1..G4`, equivariance defect (`<= 1e-6`), and M6's rank/margin candidate-Jacobian diagnostic! (Does `Arm 1 Host-Pure` ALSO extrapolate >55% past $\pi/2$ on `G1/G3`, or does frozen random projection struggle without coupled tokens / sufficient width?)
25:   - Write down the exact permutation table distinguishing: (i) **synthetic/canonical per-window shift** `(r1, r2)`, (ii) **real hardware common receive-gauge turn** `r_rx` (affecting Q1/Q2/Q3_rx), and (iii) **joint TX+RX transform**, plus exact incorporation of M4's shared CFO latent `(omega_hat, v_omega)` (via Gaussian-Hermite over $\omega$ or analytical coupled variance $n^2(v_\phi d + v_\omega d^2)$ with shared edge correlation) and M3's `joint[4,4]` factor!
```

### Event #13: `item.completed` — `command_execution` (`item_11`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command 'Get-Content -Encoding utf8 research/stage4_math_and_design/M2/reply_r1.md | Select-Object -Skip 12 -First 15; Get-Content -Encoding utf8 research/stage4_math_and_design/discussions/M2/prompt_r2.md -TotalCount 85'
```

**Aggregated Output:**

```text
Both Q3 arms observe the same signed metadata and codes. Bayes sign MI is available to both. On G2 and G4 zero retains its advantage under optimal scalar scaling, with \(0.031042\) and \(0.030809\) lower projected reverse-conditional KL. Work F's seed-mean orientation gain is 0.009820 for phase and 0.005048 for zero. Zero retains signed text. On G1 phase, holding the other path matched, flipping text costs 0.003231 nats and flipping slots costs 0.017240, with slot effects \(−0.000641,+0.035214,+0.017147\) across seeds. The within-checkpoint slot response does not establish a stable gain across retrains. On Q2 valid, recalibration reduces the phase lead from 0.021807 to 0.001594. The fresh-resample lead reverses from +0.000757 to −0.002672. All seven Q2 controls are in the full report.

**Geometric pooling preserves the failure pattern.** G1 arithmetic/geometric give phase 17.05%/16.36% at NLL 1.411312/1.411586 and zero 16.43%/16.60% at 1.439122/1.445854. G3 also loses accuracy under both pools. G2 and G4 improve accuracy and NLL with both. On complete orbits the exact arithmetic loss gain is label-weighted Jensen. For G1 zero it is 0.04120584 versus 0.03488442 for the requested reverse-KL expression. The latter uses the pooled forecast as the label law.

**Stage 4 gate.** Require absolute accuracy/NLL against uniform, dev prior and matched zero, a preregistered random-label test, and a positive linearized NIG contrast with proposed \(\eta_{\min}=0.10\). Require dev-only ECE below 0.05, paired Brier improvement of at least 0.005 against the strongest locked baseline, and maximum equivariance/idempotence error at most \(10^{-5}\). Require all endpoints within each claim and apply Holm across separate synthetic, Pluto and CTTC claims. Add a fixed-text slot intervention for claims about signed-slot use.

- **M1** should cross accumulated angle and frequency, with matched text, slots, code, waveform and physical references in the intervention design.
- **M3** should compare native forecasts with arithmetic, geometric and representative readouts on common development data and lock calibration before blind evaluation. G1 fails the absolute gate under the raw and projected scores.
- **M4** should retain 131 synthetic blocks pending full-gate pilots. With G1 variance and 3 pp/0.04 nats true excess above the margins, the historical pair needs 7/5 independent blocks for 95% marginal power conditional on the three fixed seeds. Across retrains, three seeds cap even known-variance power at 71.54%/85.94%, regardless of block count. Ten paired seeds give proxy minima of 15/7 independent blocks or captures, or 7/5 CTTC sessions of eight clusters at proxy session ICC 0.10. At the observed G1 excess, the fixed-seed capture proxy needs 10/17. Treat Pluto's 16 captures and CTTC's nine sessions as separate design populations, measure their variance on independent pilots, and power every gate endpoint before locking real-data enrollment.

The pinned CPU run exited 0. The in-scope Stage 3D summary replay has maximum absolute difference 0. [Theorem and full results](THEOREM_M2.md) · [Verifier](verify_m2.py) · [Numerical results](results_m2.json)

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

### Event #14: `item.completed` — `command_execution` (`item_12`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "Get-ChildItem -Name research/stage4_math_and_design/M4 -Force; uv python find 3.12; rg -n '"'^def |''^class '"' research/stage4_math_and_design/M4/verify_m4.py"
```

**Aggregated Output:**

```text
reply_r1.md
results_m4.json
THEOREM_M4.md
verify_m4.py
C:\qazwsx\python.exe
37:def branch_text(ref, folder, filename):
43:def literal_constants(source, names):
72:def noise(rng, shape):
76:def gaussian_kl(mu, cov, mu0, cov0):
83:def centered_t_kl(nu, m, trace_ratio, log_det_ratio):
91:def assert_mc(values, expected, sd_multiple=6):
98:def verify_distributions(rng):
144:def posterior_and_mi(weights, labels):
150:def channel_information(labels, codes, code_count):
167:def encoder_metrics(encoder, windows, labels):
184:def lift(x, ancillary):
188:def frozen_encoder_sweep(rng, encoder):
267:def nuisance_curves(rng, encoder):
353:def guarded_intervals(ranges, circumference=None):
362:def cubic_samples(center):
366:def symbol_support(k, sps, fractional, offsets, base=0):
370:def audit_local_support(unit, holdout, allowed_nuisance, refs, targets, sps, offsets):
384:def audit_splits(anchors):
487:def verify_manifests(check_http):
542:def verify_budget(splits, lock):
591:def main():
```

### Event #15: `item.completed` — `command_execution` (`item_10`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command 'Get-Content -Encoding utf8 research/stage4_math_and_design/M6/THEOREM_M6.md | Select-Object -Skip 130 -First 62; Get-Content -Encoding utf8 research/stage4_math_and_design/discussions/M6/prompt_r2.md -TotalCount 81'
```

**Aggregated Output:**

````text

With one-sided alpha $\alpha=.05$, use O'Brien–Fleming-style spending $a(t)=\bar\Phi(z_{1-\alpha}/\sqrt t)$ or predeclare Pocock-style $a(t)=\alpha\log[1+(e-1)t]$ for more early alpha. For the former, $a(.18)=.00005288548$, $b_1=3.876957$; solve $a(t)+\int_{-\infty}^{b_1}\varphi(z)\bar\Phi((b_2-\sqrt tz)/\sqrt{1-t})\,dz=.05$ for $b_2=1.645088$. Refusing continuation can only remove rejection paths if boundaries are held fixed. Componentwise monotonicity in $\mu$ puts the least favorable null at zero. Familywise guarantees require preallocated alpha across multiple promotion hypotheses, or a gatekeeping order; for mandatory jointly successful endpoints, an intersection-union gate can use the full level per component while requiring all successes. [Lan and DeMets, 1983](https://eclass.uoa.gr/modules/document/file.php/MATH301/PracticalSession3/Lan_DeMets_1983.pdf)

Under the planning prior $\mu\sim N(0,4)$, $\mu\mid Z_1=z\sim N(v\sqrt tz,v)$ with $v=(1/4+t)^{-1}$, from completing the square. The predictive distribution for $Z_2$ has mean $\sqrt tz[1+(1-t)v]$ and variance $1-t+(1-t)^2v$. Continue if predictive $P(Z_2\ge b_2\mid z)\ge .10$, equivalently $z\ge -0.270438$; early efficacy proceeds to the full rollout for independent validation. The exact least favorable Type-I rate is **.050000** with nonbinding futility and **.044695** with modeled binding futility and no CPU screen. These are probability calculations under the specified Gaussian law; the Monte Carlo has sampling error.

The paired 10,000-candidate replay assumes a 60% structural-null stratum ($\mu=-.8$, CPU pass .10), 15% boundary null ($\mu=0$, pass .90) and 25% working ($\mu=3.6$, pass .99). Pass draws are conditionally independent given stratum. These are planning assumptions, not measured rejection rates for M1–M5. Both policies see the same potential Gaussian scores. Fixed spends the archival \$17.12 envelope for every candidate. Sequential spends a modeled 8.5% if CPU passes and the pilot is bought ($.75\times.18/3+.04$ of fixed, comprising a training share and a pilot overhead share). It pays the remaining fraction on continuation including early efficacy. This scales an archival bill as a scenario; it neither prices an individual seed retrospectively nor assumes a \$17.12 charge in the Stage 4 cap. If $p_s(\mu)=P(\mathrm{CPU\ pass}\mid\mu)$, then

$$
\mathbb E[C_{\rm seq}\mid\mu]=C_Fp_s(\mu)[f+(1-f)\bar\Phi(z_f-\sqrt t\mu)].
$$

| Outcome | Fixed MC | Sequential MC | Fixed analytic | Sequential analytic |
|---|---:|---:|---:|---:|
| Boundary null false efficacy | 4.667% | 3.600% | 5.000% | 4.023% |
| Mixture of both null strata | 1.760% | 0.800% | 1.580% | 0.856% |
| Working power | 97.520% | 93.280% | 97.472% | 93.395% |
| Mean modeled cost | \$17.12 | \$6.101243 | \$17.12 | \$6.107866 |
| Saving versus fixed | — | **64.3619%** | — | **64.3232%** |

Sequential exact-binomial MC intervals are **[2.716%, 4.671%]** on the boundary-null stratum and **[92.227%, 94.230%]** on the working stratum. The modeled 90% power needs drift **≥3.267657** at .99 CPU sensitivity. At .95 sensitivity the same working stratum drops below 90% power. If every candidate is working, the analytic saving drops to about **4.27%**. These are selection and effect-size sensitivities, not a transfer claim. Savings come from declining expensive branches, not more favorable final testing.

**Actual Stage 4 release.** Use the pilot primarily for engineering, variance, and predictive futility on disjoint development mothers, sessions or blocks. Freeze pilot/final coupling and calibrate on external development simulations before using the two-look efficacy boundaries. If that law cannot be justified, allow only futility at the first look and apply a single valid, independent final test on blind blocks after all three seeds and scoring settings are fixed. Use new synthetic mother namespaces, disjoint from the Stage 3D diagnostic archives, and hold out all physical records used for the claimed final test. Conditional on the pilot and model selection, the blind test must be valid for the declared null. Its unconditional false-promotion rate is at most its level by iterated expectation. Power after futility, particularly with seed-to-seed heterogeneity, still needs development calibration. Full release also requires passing physical margins and negative controls. Count mothers or recording clusters as independent units under the registered assumptions. Use randomization inference where the experimental randomization supports it, or a declared block model with sensitivity to correlation. For Pluto, 16 temporal blocks support inference within the capture when justified. For CTTC, report 72 nested blocks and session-level (9-unit) sensitivity. A sign-flip test is exact for justified randomization or a stated joint sign-symmetry null. Mean-zero shifts alone supply neither condition.

### 4.3 Stage 4 Modal plan in US dollars

The arithmetic charges compute by billed seconds using public base rates A100-80GB .000694/s, L4 .000222/s and A10 .000306/s. All cloud lines include a planned ceiling of **2 physical cores (4 vCPU) and 16 GiB** at .0000131/core/s and .00000222/GiB/s. These are resource ceilings for the estimate. Confirm actual measured consumption fits before extending the pilot. Use standard preemptible Functions, one container at a time, no region premium, and budget cold start/scale-down in the seconds shown. The \$12.88 is a historical planning figure. Verify account usage, free storage, egress and pricing before launch. [Modal pricing](https://modal.com/pricing), [Resource controls](https://modal.com/docs/guide/resources)

| Work | Device | Max billed seconds | Rate incl. CPU/RAM | Cost cap |
|---|---|---:|---:|---:|
| Download/reference alignment, causal frontends, dataset audit and weight staging to Volume | CPU | 7,200 | .00006172 | .444384 |
| Pinned synthetic smoke, cache/version parity and timing | A100-80GB | 450 | .00075572 | .340074 |
| Synthetic pilot, 1 seed ≤173 of 960 steps (18.02%) | A100-80GB | 650 | .00075572 | .491218 |
| Resume seed 1 and train seeds 2 and 3, ≤2,707 steps total | A100-80GB | 6,750 | .00075572 | 5.101110 |
| Synthetic blind G1–G4 + Q2_C4, forward-only KV scoring | A100-80GB | 1,050 | .00075572 | .793506 |
| Pluto 850 MHz, 2,240 blind frames / 16 blocks, forward only | L4 | 2,300 | .00028372 | .652556 |
| CTTC, 1,440 blind packets / 72 blocks / 9 sessions, forward only | L4 | 1,700 | .00028372 | .482324 |
| Base compute envelope | | | | **8.305172** |
| Substitute A10G for L4 if selected before scoring, incremental price at all 4,000 seconds | A10G | | .00008400 increment | .336000 |
| Remaining reserve inside cap, including any allowed cold starts or short A100 overflow | | | | .858828 |
| **Authorization cap / buffer vs \$12.88** | | | | **9.50 / 3.38** |

Make dataset acquisition, weight pulls, cache/table computation, causal normalization, provenance linting, calibration folds, and source guard checks CPU jobs. Mount prepared read-only dataset features for evaluation and commit isolated outputs; local CPU reduces cloud seconds where practical. The three synthetic seeds train the one surviving Equivariant+RefPool design. Historical phase/zero archives, physical baselines and exact intervention controls give context. This envelope does not buy six full retrained comparator checkpoints. Any newly trained matched-baseline arm is a separate priced decision. Real evaluation uses those same frozen synthetic-trained weights and frozen candidate schema. If real development data are used to fit preprocessing or a temperature, freeze parameters by source fold and report strictly zero-shot calibration separately where claimed.

At the historical 1.9757 s/step the 2,707 continuation steps would take ~5,348 s before overhead. Require a receipt within 2.30 s/step, leaving ~524 s for all continuation overhead at that worst allowed rate. The new preprocessing, fused kernels and longest Q2 inputs can change this. At most 6,144 Q3 prefixes cover 4 cells × 2 signs × 256 exact keys × 3 seeds, provided the key includes all prefix state, dtype and batch policy. Q2's seven evidence arms require at most 10,752 prefixes for 512 mothers × 3 seeds. The 1,050 s synthetic scoring line therefore requires 16.1 prefixes/s before overhead. Cache prompts only across identical embeddings, masks, positions, template, candidate scope and attention state. Distinguish KV/linear-state sharing from lookup deduplication. Fix the padding and batch policy and require per-probability and paired endpoint parity on a receipt set before any blind inference. Historical BF16 differences across batch settings reached .067 in probabilities with per-cell argmax agreement as low as .6875. Historical A100 b64 peaked at 45,410 MB on inference, so use smaller fixed batches or prevalidated kernels on 24 GB cards. Keep precision locked across comparable arms.

The real rows cap at one query per packet/frame for up to seven evidence arms × three seeds. Without dedup, Pluto requires 47,040 prefixes (≥20.46/s within 2,300 s) and CTTC 30,240 (≥17.79/s within 1,700 s). Reserve startup separately within each row before setting a required steady rate. These rates, and VRAM on L4/A10G, require a parity-and-throughput witness under the actual workload. Use part of the relevant row for an unblinded developer smoke. Select the lowest-cost parity-passing device whose worst-case authorized count fits. Recompute cost after any failed receipt. Substitution allowance covers price difference at equal seconds, while a first device's already billed time still counts. A100 fallback is possible only for a separately metered subset within remaining cost/time. If one resource proof fails, preserve the CPU-only evidence and mark that GPU row incomplete. Neither the time cap nor the historical A100 rate mathematically guarantees real workload completion. Keep per-invocation deadlines, minimal scale-down, charge for billed teardown and any retries, and precharge each next shard at its bounded worst-case dollars against a single serial ledger. Account for already running and settling time before dispatch. Modal dashboard budget enforcement can lag and availability varies by plan. [Volume staging](https://modal.com/docs/guide/volumes), [Modal budgets](https://modal.com/docs/guide/budgets)

## 5. Stage 4 → 6 critical path

```mermaid
flowchart TD
    A[4A Causal data manifest and physical controls] --> B[4B Cache risk brackets and collision tests]
    B --> C[4C Equivariant RefPool and host Jacobian]
    C --> D[4D Capped 4B synthetic pilot and blind evaluation]
    D --> P[4E Pluto blind external prediction]
    D --> T[4F CTTC held-session prediction]
    P --> E[5A Trajectory and jump identifiability]
    T --> E
    E --> F[5B Event model and interval validation]
    F --> G[6A Active read and multi-turn 4B]
    G --> H[6B Qwen3.5-9B portability]
    H --> I[6C Qwen3.8-27B portability]
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
````

### Event #16: `item.completed` — `web_search` (`exec-8090bf44-80f3-4f98-b1ca-6ae4bb901892`)

```json
{
  "id": "exec-8090bf44-80f3-4f98-b1ca-6ae4bb901892",
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
      "ref_id": "turn3view0",
      "snippet": "Total lines: 391",
      "thumbnail_url": "https://images.openai.com/static-rsc-1/A4WRp4TRAah0d_mz0L1NMIoJYaxvoVXGXMpE2D8o1vpM7WdhpA8hDpkpxoP0E6rZ_z4bWDMy7sw_3JsXyPevRQ",
      "title": "Pricing: Pay per second for GPUs and CPUs | Modal",
      "url": "https://modal.com/pricing"
    },
    {
      "type": "text_result",
      "domain": "modal.com",
      "ref_id": "turn3view1",
      "snippet": "Total lines: 51",
      "thumbnail_url": "https://images.openai.com/static-rsc-1/DP4TsPys10gls19Inp-cexs_3xEH_LpofAvNBwD8XiGyHrPy-njvPQTstUTG5Lhfkxp6unR8K18JBlf_aIWKBxbjB5Q8G8aCwqY2ZCD7bqM3QCvbq7uOUr4ifZDk10aN",
      "title": "Budgets | Modal Docs",
      "url": "https://modal.com/docs/guide/budgets"
    },
    {
      "type": "text_result",
      "domain": "modal.com",
      "ref_id": "turn3view2",
      "snippet": "Total lines: 391",
      "thumbnail_url": "https://images.openai.com/static-rsc-1/3HkQlB65Vx9BXbpuvemg5bJztXNoAyfidnU_Cuab0fIXJdqWBaPu9CJd50e-9XoWDc8geXfxDOt6EBmIHKL80w",
      "title": "Pricing: Pay per second for GPUs and CPUs | Modal",
      "url": "https://modal.com/pricing"
    },
    {
      "type": "text_result",
      "domain": "modal.com",
      "ref_id": "turn3view3",
      "snippet": "Total lines: 84",
      "thumbnail_url": "https://images.openai.com/static-rsc-1/hy5lCJ2onE71Rgfqkot5aEEX0EZOZsXTC7QgTR6pKTBZCsrMsaMc2a98z6yD9S4CTWvNAO6RyEjadJN-9xEsvnuq8YlQ4x4ZC2IVCRDMYYxPmgruT9JueGHnxpY-KrR5pOE0bqhQNrIcolPsOtH7FLVpsrGK0Y97Flt7Cgh_GUc",
      "title": "Configuring CPU, memory, and disk | Modal Docs",
      "url": "https://modal.com/docs/guide/resources"
    }
  ]
}
```

### Event #17: `item.completed` — `command_execution` (`item_15`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command 'Get-Content -Encoding utf8 research/stage4_math_and_design/discussions/M3/prompt_r2.md | Select-Object -Skip 24 -First 30'
```

**Aggregated Output:**

```text
3. Add deterministic CPU unit/property verifications inside `verify_m3.py` (keep total runtime concise `< 60s` or reuse cached fast quadrature; verify: (i) zero double-counting invariant against prior state, (ii) non-trivial shared CFO + Brownian multi-clock gain vs naive sum, (iii) byte-exact `uint8`/`float32` representation identity across shuffled/duplicate/distractor/padded `RefPool` inputs, and (iv) Stage 6 pre-reveal predictive marginal + post-reveal Bayes update consistency!). Run via `uv run` and report in `reply_r2.md`.
```

### Event #18: `item.completed` — `command_execution` (`item_16`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "Get-Content -Encoding utf8 research/stage4_math_and_design/M4/verify_m4.py | Select-Object -Skip 535 -First 107; rg -n 'T_CAND|f708' research/stage4_math_and_design/M4/THEOREM_M4.md"
```

**Aggregated Output:**

```text
            except OSError as error:
                status[name] = {'metadata_reachable': False, 'detail': str(error)}
    return {'pluto': p, 'cttc': c, 'total_download_bytes': p['size_bytes'] + c['all_selected_download_bytes'],
            'live_http': status}


def verify_budget(splits, lock):
    report = LOCK_PATH.with_name('REPORT.md').read_text(encoding='utf-8')
    for string in ('$17.12', '$30.00', '$12.88', '$0.00', '1,975.7', '23.88', '52,972.5', '45,409.9'):
        assert string in report
    arms, steps, train_batch = 4, 384, 32
    rate = {'A100-80GB': Decimal('0.000694'), 'A100-40GB': Decimal('0.000583'),
            'L4': Decimal('0.000222'), 'A10': Decimal('0.000306'), 'CPU': Decimal(0)}
    cpu, mem = Decimal('0.0000131'), Decimal('0.00000222')
    real = splits['pluto']['eval_counts']['test'] + splits['cttc']['counts']['test_target']
    repeats = splits['pluto']['repeat_candidate_rows'] + splits['cttc']['repeat_candidate_rows']
    stress = splits['pluto']['stress_rows'] + splits['cttc']['stress_rows']
    synthetic = 32 * 64
    eval_counts = {'real_raw': arms * (real * 4 + repeats * 2), 'real_stress_q2': arms * stress,
                   'synthetic_blind': arms * synthetic * 5, 'dev_calibration': arms * 4000}
    assert eval_counts == {'real_raw': 82560, 'real_stress_q2': 8240, 'synthetic_blind': 40960, 'dev_calibration': 16000}
    training = arms * steps * Decimal('2.5')
    eval_seconds = {key: math.ceil(value * Decimal('0.030')) for key, value in eval_counts.items()}
    tasks = [('connector_train', 'A100-80GB', int(training), 2, 12)]
    tasks += [(key, 'A100-80GB', seconds, 2, 12) for key, seconds in eval_seconds.items()]
    tasks += [('gpu_smoke_boot_and_commit', 'A100-80GB', 1100, 2, 12),
              ('gpu_retry_reserve', 'A100-80GB', 900, 2, 12),
              ('ingest_and_verify', 'CPU', 1800, 1, 8), ('frontend_and_physical', 'CPU', 7200, 4, 16),
              ('audit_and_aggregation', 'CPU', 1800, 2, 8)]
    rows = []
    total = Decimal(0)
    for label, gpu, seconds, cores, gib in tasks:
        usd = seconds * (rate[gpu] + cpu * cores + mem * gib)
        rows.append({'phase': label, 'resource': gpu, 'seconds_cap': seconds, 'physical_cores_cap': cores,
                     'gib_cap': gib, 'usd_cap': str(usd)})
        total += usd
    free, past, target = Decimal('30.00'), Decimal('17.12'), Decimal('10.00')
    # Independent identity across all time caps, then a locked total for accidental scope changes.
    gpu_seconds = sum(row[2] for row in tasks if row[1] != 'CPU')
    assert total == gpu_seconds * rate['A100-80GB'] + cpu * sum(row[2] * row[3] for row in tasks) + mem * sum(row[2] * row[4] for row in tasks)
    assert gpu_seconds == 10274 and total == Decimal('8.44073416') and total < target < free - past
    return {'official_pricing_url': 'https://modal.com/pricing', 'usd_per_second': {k: str(v) for k, v in rate.items()},
            'physical_core_usd_per_second': str(cpu), 'gib_usd_per_second': str(mem), 'rows': rows,
            'stage3d_model_revision': lock['hf_sha'], 'stage3d_host_sha256': lock['host_sha256'],
            'stage3d_locked_training': lock['locked_hyperparameters'],
            'stage3d_run_wall_seconds': [2884.0, 3026.2, 3217.8, 2908.7, 3059.1, 2983.0],
            'stage3d_observed': {'training_seconds_per_step': 1.9757, 'eval_seconds_per_row': 0.02388,
                                 'train_peak_mb': 52972.5, 'eval_peak_mb': 45409.9},
            'stage4_eval_rows': eval_counts, 'stage4_training_examples_per_arm': steps * train_batch,
            'stage4_a100_seconds_cap': gpu_seconds, 'stage4_total_usd_cap': str(total),
            'stage3d_free_at_report_usd': str(free - past), 'stage3d_reported_billed_usd': '0.00',
            'headroom_to_ten_usd': str(target - total), 'headroom_to_historical_remaining_usd': str(free - past - total),
            'active_balance_or_modal_execution_verified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-http', action='store_true', help='Optionally probe small public metadata endpoints.')
    args = parser.parse_args()
    assert sys.version_info[:2] == (3, 12) and np.__version__ == '2.3.5' and scipy.__version__ == '1.17.0'
    lock = json.loads(LOCK_PATH.read_text(encoding='utf-8'))
    digest = hashlib.sha256(WEIGHT_PATH.read_bytes()).hexdigest()
    assert digest == lock['encoder_sha256']
    encoder = load(WEIGHT_PATH)
    assert encoder.equivariant and encoder.table.shape == (16, 4) and encoder.centers.shape == (4, 4)
    print('Verified frozen s17 identity and pinned Python/NumPy/SciPy', flush=True)
    anchors = json.loads(branch_text(CTTC_REF, 'realdata_cttc_qpsk', 'results/capture_anchors.json'))
    splits = audit_splits(anchors)
    manifests = verify_manifests(args.check_http)
    budget = verify_budget(splits, lock)
    print('Verified static splits, historical manifests and Modal arithmetic', flush=True)
    rng = np.random.default_rng(SEED)
    distributions = verify_distributions(rng)
    sweep = frozen_encoder_sweep(rng, encoder)
    curves = nuisance_curves(rng, encoder)
    result = {'status': 'pass', 'seed': SEED, 'python': sys.version.split()[0], 'numpy': np.__version__,
              'scipy': scipy.__version__, 'scope': 'synthetic CPU replay, historical metadata and prospective budget',
              'encoder_sha256': digest, 'nominal_complex_noise_variance': V0,
              'ref_commits': {ref: subprocess.check_output(['git', 'rev-parse', ref], cwd=ROOT, text=True).strip()
                              for ref in (PLUTO_REF, CTTC_REF)},
              'distributions': distributions, 'frozen_encoder_sweep': sweep, 'nuisance_curves': curves,
              'index_audit': splits, 'dataset_manifests': manifests, 'modal_budget': budget}
    path = HERE / 'results_m4.json'
    staged = HERE / '.results_m4.json.tmp'
    staged.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    staged.replace(path)
    print(f'PASS {len(sweep["cells"])} mapping cells, {len(curves["conditional_curves"])} nuisance cells, {path.name}')


if __name__ == '__main__':
    main()
125:The primary samples retain actual fading, residual DC/ISI, noise, clock offset, and any changes present at registered times. A naturally occurring jump is a stratum, with changes flagged by a predeclared causal split-local pilot innovation criterion and a retrospective scorer-only morphology check. A CPU-only cross-frame Q1 diagnostic pairs `(f−1,f)` at `f mod 20 = 0` within each test block, using the earlier frame's \(k_1\) and current frame's \(k_1\) from the table. Its matched control pairs the preceding two interior frames `(f−2,f−1)` in that same block. Use only available sync and authorized pilots and a gap-class prior when the time between Pluto columns is unavailable. This samples recorded batch boundaries by schedule without requiring or asserting a jump at each boundary. Separate source domains and nominal session strata in analysis, keeping capture IDs out of model text/features. Hold `f708_snr10` out of fitted connector weights, temperatures and shared prior tuning to expose one capture transfer; allow its own causal sync and released pilots. Pluto is a blocked same-campaign test. CTTC has one public cyclic TX sequence; input-source disjointness and capture dependence are distinct issues. Use block-paired NLL/SER differences on 16 Pluto groups and 72 CTTC session-loops, group adjacent repeat pairs for CTTC Q3, and report a capture-level sensitivity analysis and source-ordinal sensitivity because loops/captures reuse source positions. Report all registered rows and lock coverage, with the same uniform predictive fallback on failed rows and a secondary locked-only analysis. Avoid treating all packets as independent. Select the Q2 `two−one` proper-loss contrast and large-angle synthetic Q3 coupled-vs-additive contrast before test, with paired intervals and the multiplicity rule frozen with M2.
```

### Event #19: `item.completed` — `command_execution` (`item_14`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command 'Get-Content -Encoding utf8 research/stage4_math_and_design/M5/verify_m5.py -TotalCount 155; Get-Content -Encoding utf8 research/stage4_math_and_design/M5/THEOREM_M5.md | Select-Object -Skip 114 -First 14'
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

```

### Event #20: `item.completed` — `command_execution` (`item_17`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "Get-Content -Encoding utf8 research/stage4_math_and_design/M2/THEOREM_M2.md | Select-Object -Skip 74 -First 38; rg -n 'capture_id' research/stage4_math_and_design/M4/verify_m4.py"
```

**Aggregated Output:**

```text
\[
L(q)-L(Pq)=\mathbb E_X\sum_y p_y(X)J_y(X)\ge0.
\]

Equality holds exactly when, almost surely, every label of positive posterior mass has the same probability across every aligned member. If the posterior has full support, this is equivariance almost surely. On an arbitrary finite design the exact formula adds \(L(q)-|G|^{-1}\sum_g L(R_gq)\) to the label-weighted Jensen gap. The complete orbits constructed from each saved mother have an invariant design by construction. Saved rotations are an incomplete orbit, so the restriction matters.

The proposed \(|G|^{-1}\sum_g D_{\rm KL}(Pq\Vert R_gq)\) is \(\sum_y m_y J_y\) pointwise. It corresponds to self-weighting by the pool. It equals the log-loss gain if \(\mathbb E[(p-m)\cdot J]=0\), in particular if \(p=m\). It is not a general equality even under exact invariance. For an invariant binary example, take two equiprobable observations interchanged by a group of order two with identity label action and \(p=(1,0)\) on both. The forecasts are \((0.8,0.2)\) and \((0.4,0.6)\). The pool is \((0.6,0.4)\), and \(J_1=\log(0.6/\sqrt{0.32})\), \(J_2=\log(0.4/\sqrt{0.12})\) differ. The loss gain equals \(J_1\). The reverse-KL mean equals \(0.6J_1+0.4J_2\).

For geometric pooling let \(h_y=\exp(|G|^{-1}\sum_g\log(R_gq)_y)\), \(Z_h=\sum_y h_y\), and \(q_{\rm geo}=h/Z_h\). Componentwise AM–GM gives \(Z_h\le1\), and

\[
L(q_{\rm geo})=|G|^{-1}\sum_g L(R_gq)+\mathbb E_X\log Z_h.
\]

Thus both pools weakly improve invariant-design log risk over the raw model. Their ordering depends on the label law and the forecasts. Normalization makes the geometric pool the minimizer of \(\sum_g D_{\rm KL}(r\Vert R_gq)\) over \(r\). It is equivariant, and applying it again returns the same distribution. The arithmetic pool minimizes \(\sum_g D_{\rm KL}(R_gq\Vert r)\) and is also equivariant and idempotent.

For each target \(y\), the closed top-1 region is the intersection of halfspaces \(r_y-r_k\ge0\) over \(k\ne y\). For the arithmetic path, the exact margin to competitor \(k\) at the barycenter is the mean of aligned margins. A unique raw correct decision is lost iff a mean margin is negative or hits a tie boundary. A unique correct decision is gained iff all mean margins are positive after a raw error or tie. The fractional-tie hit function at a row is

\[
h_y(r)=\frac{1[\min_{k\ne y}(r_y-r_k)\ge0]}{|\arg\max_j r_j|}.
\]

Over a target distribution, strict accuracy decline is equivalent to \(\mathbb E[(h_y(q)-h_y(Pq))_+]>\mathbb E[(h_y(Pq)-h_y(q))_+]\), including ties. At population level this is the same as \(\mathbb E_X[p(X)\cdot(d(q)-d(Pq))]>0\), where \(d\) is uniform on the maximizer set. For geometric pooling the corresponding competitor margins are averages of \(\log r_y-\log r_k\). Log score depends on the probability assigned to the true class, while top-1 depends on which side of the decision planes the pool lies.

## 3. Stage 4 frozen statistical gate

A claim passes when every endpoint for that claim clears its limit. Freeze the candidate, seed handling, allowed observation sets, per-class weighting, code below, multiplicity family, score floor, calibration, and all margins before unblinding. Apply the same three gate parts separately to the synthetic extrapolation claim, the Pluto claim and the CTTC claim. For a claim covering all four synthetic cells, require each cell to pass. Report each failed or indeterminate endpoint. A winner picked from several candidates requires a design split or inclusion in the multiplicity family.

1. **Absolute and comparative performance.** Use paired lower bounds. Require NLL to beat uniform, the dev-frozen prior, and the matched zero-phase system, with a \(0.02\)-nat increment proposed against matched zero. Require accuracy above analytic chance, a dev-frozen prior classifier, and the matched zero-phase system, with \(+0.02\) absolute over chance and zero proposed as initial margins. With imbalanced classes, also require outperformance of the fixed majority prediction or used prior competitor, and report balanced accuracy as a separately locked estimand. Require valid chance/random-label randomization tails for the accuracy and NLL statistics where the design supplies exchangeability. Choose an oracle with the same information budget as the candidate for the main NIG denominator and disclose a finer physical oracle separately. Propose \(\eta_{\min}=0.10\) and validate \(H(Y)-L(o)>0\) on an independent development design or by a co-primary lower bound. Test the linear contrast \(D_{\eta}=(1-\eta)H(Y)-L(q)+\eta L(o)>0\) rather than dividing two noisy quantities. Use \(H(Y)=\log K\) for a genuinely uniform target law. Otherwise predefine the target distribution or carry uncertainty in \(H(Y)\) by a joint confidence region and minimize the contrast over it. For real data without a defensible Bayes oracle, freeze a physically informed predictive reference, call the resulting metric reference-normalized, and withhold the Bayes-normalized claim. Requiring a learner to outperform a true Bayes oracle under proper loss is infeasible. A positive NIG floor defines a fraction of the attainable improvement. Controlled slot/text interventions form a further prerequisite for a specifically signed-slot mechanism claim. Require improvement with matched versus permuted or flipped slots at fixed text and same \(X\), with seeded training and ablations paired. Use the native receiver outputs for the gate.
2. **Calibration and structure.** Propose an upper bound of 0.05 on top-label ECE after dev-only calibration with ten frozen bins. Propose a paired Brier improvement of at least 0.005 against the strongest locked trivial or zero-phase Brier competitor, plus a strict improvement over every required comparator. For \(K=4\) uniform Brier is \(1-1/K=0.75\) with the unnormalized multiclass sum. Prelock a probability transformation ensuring finite NLL, for example \(q_\epsilon=(1-K\epsilon)q+\epsilon\) with small fixed \(\epsilon\), and score that forecast consistently. For calibration inference set \(d_j=\mathbb E[1_{\{\operatorname{conf}\in I_j\}}(h_Y-\operatorname{conf})]\); \(\operatorname{ECE}=\sum_j|d_j|=\max_{v\in\{-1,1\}^{10}}v^\top d\). Test all \(2^{10}\) linear upper-margin conditions as an IUT or use a valid joint confidence region for \(d\). Estimating the absolute value first and treating empirical ECE as an unbiased, normal per-mother loss has no exact justification. Require \(\max_{g,x}\|q(gx)-\rho(g)q(x)\|_\infty\le10^{-5}\) and projection/readout idempotence \(\max_x\|P(Pq)(x)-Pq(x)\|_\infty\le10^{-5}\). Verify across the whole finite code domain and all group elements on synthetic. For continuous real inputs distinguish formal architectural guarantees from the observed maximum on prespecified transforms. Test only valid actions with physically correct label permutations. Audit the deployed path and the offline projector separately.
3. **Hierarchy and multiplicity.** Keep all repeats, rotations, acquisitions, and matched arms inside their mother and higher cluster. Report a frozen-seed conditional estimate and, if claiming generalization to retraining, a separate random-seed inference. Use equal independent cluster weights for the following exact balanced analysis. Pluto supplies 16 units if they are independent captures. For a CTTC new-session claim use nine independent session means, with all eight clusters per session contributing to that session. A conditional claim on the fixed sessions needs the corresponding estimand and defensible within-session cluster independence. For a permutation or label-randomization gate, lock the allowed exchangeability orbit and recompute the same statistic throughout. Repeated/rotated labels move jointly with their mother and never split correlated session units as though independent. A synthetic randomized design with independent label assignments can instead rerandomize the sanctioned generator operation inside the original design. When there is no justified exchangeability, collect a randomized negative control. State whether it tests a sharp random-label null or a mean-effect margin. A permutation p-value \((1+\#\{T_{\rm perm}\ge T_{\rm obs}\})/(R+1)\) with uniform randomizations and ties included is conservative under the sharp design. Use exact enumeration when feasible.

### Exact type-I control and power under a balanced Gaussian model

For an oriented block contrast \(D_{bs}\) let \(D_{bs}=\mu+u_b+\nu_s+e_{bs}\), with independent Gaussian families of variances \(\sigma_u^2,\sigma_\nu^2,\sigma_e^2\). These are the assumptions for the exact parametric calculations. Blocks already aggregate mothers and correlated views. If \(m\) mothers in a block have common variance \(v\) and exchangeable within-block correlation \(\rho\), a block mean contributes \(v[1+(m-1)\rho]/m\); this design effect applies to the particular contrast and variance component being modeled. The Stage 3D Work E lock estimated block SD on a fixed-seed pilot, took the larger SD from pilot upper bound and design-effect scenario and used the conditional noncentral t. It did not price uncertainty across retrains.

For \(B\) blocks and \(S\) seeds,

459:                    uid = (anchor['capture_id'], loop, sp, role, k)
```

### Event #21: `item.completed` — `command_execution` (`item_18`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "git grep -n -i -e 'sample rate' -e 'symbol rate' -e '2e6' -e '1e6' origin/research/realdata-pluto-qpsk -- research/realdata_pluto_qpsk/PROTOCOL.md research/realdata_pluto_qpsk/REPORT.md; Get-ChildItem -Name research/stage4_math_and_design/M1,research/stage4_math_and_design/M2,research/stage4_math_and_design/M3,research/stage4_math_and_design/M6; Get-Content -Encoding utf8 research/stage4_math_and_design/discussions/M3/prompt_r2.md -TotalCount 65"
```

**Aggregated Output:**

```text
origin/research/realdata-pluto-qpsk:research/realdata_pluto_qpsk/REPORT.md:70:- **留出探针 `[21:26]` 隔离验证**：向探针注入 `1e6 - 1e6j` 污染值，全部在线库与 13 条接收臂的信道/载荷输出最大差值 **`0.0`**（比特精确零泄漏）；
.uv-cache
.uv-tmp
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
results_m6.json
THEOREM_M6.md
verify_m6.py
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
```

### Event #22: `item.completed` — `command_execution` (`item_19`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "Get-Content -Encoding utf8 research/stage4_math_and_design/M5/verify_m5.py | Select-Object -Skip 157 -First 70; git show origin/research/realdata-cttc-indoor-qpsk:research/realdata_cttc_qpsk/results/capture_anchors.json | rg 'capture_id'; Get-ChildItem -Name research/stage4_math_and_design/M6"
```

**Aggregated Output:**

```text
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


def polar_emissions(angle_bits, radial_bits, mean_amp, noise_ratio, order=18, radial_boundaries=None):
    """Integrate the complex Gaussian probability of each packed polar cell."""
    node, weight = np.polynomial.legendre.leggauss(order)
    # Scaled radii farther than 11 standard complex deviations have negligible mass.
    edges = radial_edges(radial_bits) if radial_boundaries is None else radial_boundaries
    r_edge = edges / np.sqrt(noise_ratio)
    r_edge = np.clip(r_edge, max(0., mean_amp - 11), mean_amp + 11)
    angles = 4 * 2**angle_bits
    width = 2 * np.pi / angles
    symmetry = gcd(PHASES, angles)
    base_phases = PHASES // symmetry
    angle_shift = angles // symmetry
    result = np.zeros((base_phases, angles, len(r_edge) - 1))
    phase = np.angle(UNIT[:base_phases])[:, None, None]
    radial_node = (r_edge[:-1, None] + r_edge[1:, None] + (r_edge[1:, None] - r_edge[:-1, None]) * node) / 2
    radial_weight = (r_edge[1:, None] - r_edge[:-1, None]) * weight * radial_node / 2
    subdivisions = max(1, int(np.ceil(mean_amp * width / 2)))
    shifts = ((np.arange(subdivisions)[:, None] + (1 + node) / 2) / subdivisions).ravel()
    angular_node = -np.pi / 4 + (np.arange(angles)[:, None] + shifts) * width
    angular_weight = np.tile(weight, subdivisions) * width / (2 * np.pi * subdivisions)
    for k, (r, rw) in enumerate(zip(radial_node, radial_weight)):
        # Subdivide broad radial or angular bins for concentrated, high-SNR profiles.
        if (r_edge[k + 1] - r_edge[k]) > 2:
            edges = np.linspace(r_edge[k], r_edge[k + 1], 9)
            r = ((edges[:-1, None] + edges[1:, None]) / 2 + np.diff(edges)[:, None] * node / 2).ravel()
            rw = (np.diff(edges)[:, None] * weight / 2).ravel() * r
        distance = (r - mean_amp)**2 + 2 * r * mean_amp * (1 - np.cos(angular_node - phase))[..., None]
        result[:, :, k] = (np.exp(-distance) @ rw) @ angular_weight
    np.testing.assert_allclose(result.sum(axis=(1, 2)), 1., atol=5e-6, rtol=0)
    base = np.log(np.maximum(result, 1e-280))
    return np.concatenate([np.roll(base, shift * angle_shift, axis=1) for shift in range(symmetry)])


def window_log_likelihood(method, t, residual, packets, bins, a, v, encoder, key):
    count = len(t)
    values = np.empty((count, len(OLD), 2, PHASES))
    emissions = {}
    for window in range(len(OLD)):
        for noise_id, multiplier in enumerate((1., 6.)):
            variance = v[window] * multiplier
            if method in ("full_iq", "canonical_continuous"):
                # Common constants independent of the trajectory cancel in Bayes.
                distance = abs(t[:, window, None] - a[window] * UNIT)**2
                values[:, window, noise_id] = (
                    -(distance + residual[:, window, None]) / variance - 3 * np.log(variance)
                )
            elif method == "4_s17":
                condition = (a[window], variance)
                if condition not in emissions:
    "capture_id": "f2450_snr20_QPSK",
    "capture_id": "f2450_snr10_QPSK",
    "capture_id": "f2450_snr0_QPSK",
    "capture_id": "f708_snr20_QPSK",
    "capture_id": "f708_snr10_QPSK",
    "capture_id": "f708_snr0_QPSK",
    "capture_id": "f433_snr20_QPSK",
    "capture_id": "f433_snr10_QPSK",
    "capture_id": "f433_snr0_QPSK",
reply_r1.md
results_m6.json
THEOREM_M6.md
verify_m6.py
```

### Event #23: `item.completed` — `command_execution` (`item_20`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$src='"'C:/Users/李则徐/AppData/Local/uv/cache'; "'$dst='"'research/stage4_math_and_design/M4/.uv-m4-r2'; New-Item -ItemType Directory -Force -Path \""'$dst/archive-v0","$dst/wheels-v6/pypi","$dst/simple-v21/pypi" | Out-Null; foreach ($p in '"'3Oxde2-Jkv6KGjnQ','cSnmXl2iWBpNAXGE') { Copy-Item -LiteralPath \""'$src/archive-v0/$p" -Destination "$dst/archive-v0" -Recurse }; foreach ($p in '"'numpy','scipy') { Copy-Item -LiteralPath \""'$src/wheels-v6/pypi/$p" -Destination "$dst/wheels-v6/pypi" -Recurse; Copy-Item -LiteralPath "$src/simple-v21/pypi/$p.rkyv" -Destination "$dst/simple-v21/pypi" }'
```

### Event #24: `item.completed` — `file_change` (`item_21`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M4\\verify_m4.py",
    "kind": "update"
  }
]
```

### Event #25: `item.completed` — `agent_message` (`item_22`)

The pricing difference comes from host RAM. M6’s 16 GiB allocation adds $0.00000888 per second over M4’s 12 GiB. I’ll use the 16 GiB rate and charge the 173-step pilot within the Host-Pure arm’s 384-step budget. The packet’s residual-quality field will come from measured taps before the R3 lift, so it reflects the radio evidence.

### Event #26: `item.completed` — `file_change` (`item_23`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M4\\verify_m4.py",
    "kind": "update"
  }
]
```

### Event #27: `item.completed` — `command_execution` (`item_24`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n '2\\.[0-9]{3}|b2|.270438|def.*futility' research/stage4_math_and_design/M6/THEOREM_M6.md research/stage4_math_and_design/M6/verify_m6.py | Select-Object -First 28; rg -n '0.3119' research/stage4_math_and_design/M5/results_m5.json | Select-Object -First 2"
```

**Aggregated Output:**

```text
research/stage4_math_and_design/M6/THEOREM_M6.md:46:| G3 angle extrapolated | 512 | 61.914% / .953916 | 62.109% / .979568 | 22.678% / 1.441047 | 19.737% / 1.478130 | .025652 | .461480 | .498563 |
research/stage4_math_and_design/M6/THEOREM_M6.md:62:\* Duplicate-raw uses the existing deduplicated Work A row as its information-equivalent physical reference. Q2 contains 512 mothers at (+0.35), without a paired Full-IQ score in the NPZ. Work A is a cache/physical reference. The host receives s17 window codes and 12-dimensional continuous reference features. Fresh evidence improves the Work A NLL by **.156017** and worsens phase NLL by **.016758**. The raw duplicate worsens phase accuracy by **2.474 pp** (mean TV **.040377**), while the dedup arm has zero TV. Two tokens for one acquisition and tokens for two genuine acquisitions test different physics. Both the actual length change and the added attention content can matter. A matched padding/mask experiment is needed to identify their separate effects.
research/stage4_math_and_design/M6/THEOREM_M6.md:134:Under the planning prior $\mu\sim N(0,4)$, $\mu\mid Z_1=z\sim N(v\sqrt tz,v)$ with $v=(1/4+t)^{-1}$, from completing the square. The predictive distribution for $Z_2$ has mean $\sqrt tz[1+(1-t)v]$ and variance $1-t+(1-t)^2v$. Continue if predictive $P(Z_2\ge b_2\mid z)\ge .10$, equivalently $z\ge -0.270438$; early efficacy proceeds to the full rollout for independent validation. The exact least favorable Type-I rate is **.050000** with nonbinding futility and **.044695** with modeled binding futility and no CPU screen. These are probability calculations under the specified Gaussian law; the Monte Carlo has sampling error.
research/stage4_math_and_design/M6/THEOREM_M6.md:150:Sequential exact-binomial MC intervals are **[2.716%, 4.671%]** on the boundary-null stratum and **[92.227%, 94.230%]** on the working stratum. The modeled 90% power needs drift **≥3.267657** at .99 CPU sensitivity. At .95 sensitivity the same working stratum drops below 90% power. If every candidate is working, the analytic saving drops to about **4.27%**. These are selection and effect-size sensitivities, not a transfer claim. Savings come from declining expensive branches, not more favorable final testing.
8827:                "mean": -0.00033119485906038494,
```

### Event #28: `item.completed` — `command_execution` (`item_25`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "Get-Content -Encoding utf8 research/stage4_math_and_design/M6/THEOREM_M6.md | Select-Object -Skip 128 -First 7; rg -n '1\\.65|1\\.69|.270|brentq|def .*seq' research/stage4_math_and_design/M6/verify_m6.py"
```

**Aggregated Output:**

```text

Use a preregistered directional **primary** score $\mu>0$ in final-standard-error units above the practical margin. In the canonical independent-increment Gaussian experiment, $Z_1\sim N(\sqrt t\mu,1)$ and $Z_2=\sqrt tZ_1+\sqrt{1-t}V\$, $V\sim N(\sqrt{1-t}\mu,1)$ independent. $t=.18$ is an information fraction. A one-seed pilot at 15–20% of optimizer steps does not, by itself, have this information fraction or this joint distribution.

With one-sided alpha $\alpha=.05$, use O'Brien–Fleming-style spending $a(t)=\bar\Phi(z_{1-\alpha}/\sqrt t)$ or predeclare Pocock-style $a(t)=\alpha\log[1+(e-1)t]$ for more early alpha. For the former, $a(.18)=.00005288548$, $b_1=3.876957$; solve $a(t)+\int_{-\infty}^{b_1}\varphi(z)\bar\Phi((b_2-\sqrt tz)/\sqrt{1-t})\,dz=.05$ for $b_2=1.645088$. Refusing continuation can only remove rejection paths if boundaries are held fixed. Componentwise monotonicity in $\mu$ puts the least favorable null at zero. Familywise guarantees require preallocated alpha across multiple promotion hypotheses, or a gatekeeping order; for mandatory jointly successful endpoints, an intersection-union gate can use the full level per component while requiring all successes. [Lan and DeMets, 1983](https://eclass.uoa.gr/modules/document/file.php/MATH301/PracticalSession3/Lan_DeMets_1983.pdf)

Under the planning prior $\mu\sim N(0,4)$, $\mu\mid Z_1=z\sim N(v\sqrt tz,v)$ with $v=(1/4+t)^{-1}$, from completing the square. The predictive distribution for $Z_2$ has mean $\sqrt tz[1+(1-t)v]$ and variance $1-t+(1-t)^2v$. Continue if predictive $P(Z_2\ge b_2\mid z)\ge .10$, equivalently $z\ge -0.270438$; early efficacy proceeds to the full rollout for independent validation. The exact least favorable Type-I rate is **.050000** with nonbinding futility and **.044695** with modeled binding futility and no CPU screen. These are probability calculations under the specified Gaussian law; the Monte Carlo has sampling error.

14:from scipy.optimize import brentq
257:def sequential_design():
273:    final = brentq(lambda b: alpha1 + final_reject(b, 0.0, -np.inf) - alpha, 1.5, 2.0, xtol=1e-14)
331:    drift_min = brentq(lambda d: 0.99 * (norm.sf(early - rho * d) + final_reject(final, d, futility_z)) - 0.90, 0, 8)
```

### Event #29: `item.completed` — `file_change` (`item_26`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M4\\verify_m4.py",
    "kind": "update"
  }
]
```

### Event #30: `item.completed` — `command_execution` (`item_28`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "Get-Content -Encoding utf8 research/stage4_math_and_design/M4/THEOREM_M4.md | Select-Object -Skip 145 -First 60; rg -n '"'^#'"' research/stage4_math_and_design/M6/THEOREM_M6.md research/stage4_math_and_design/M1/THEOREM_M1.md research/stage4_math_and_design/M3/THEOREM_M3.md | Select-Object -Last 25"
```

**Aggregated Output:**

````text

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
research/stage4_math_and_design/M3/THEOREM_M3.md:1:# M3 | Pilot sufficiency and fixed-slot pooling
research/stage4_math_and_design/M3/THEOREM_M3.md:3:## 1. Stage 3 release audit
research/stage4_math_and_design/M3/THEOREM_M3.md:11:## 2. Theorem 1 | Same-clock sufficiency and minimality
research/stage4_math_and_design/M3/THEOREM_M3.md:39:### Exact Q2 posterior
research/stage4_math_and_design/M3/THEOREM_M3.md:71:## 3. Theorem 2 | Distinct clocks require a Wiener bridge
research/stage4_math_and_design/M3/THEOREM_M3.md:111:### Error of a common-phase approximation
research/stage4_math_and_design/M3/THEOREM_M3.md:122:### Q3_after_pilot posterior
research/stage4_math_and_design/M3/THEOREM_M3.md:136:## 4. Stage 3D reference-fusion diagnosis
research/stage4_math_and_design/M3/THEOREM_M3.md:158:## 5. Fixed-slot interface and guarantees
research/stage4_math_and_design/M3/THEOREM_M3.md:173:## 6. CPU replay and paired experiments
research/stage4_math_and_design/M6/THEOREM_M6.md:1:# M6 Round 1 • CPU certificates and a capped Stage 4 experiment
research/stage4_math_and_design/M6/THEOREM_M6.md:5:## 1. The exact population loss ladder
research/stage4_math_and_design/M6/THEOREM_M6.md:38:### What the Stage 3D archive identifies
research/stage4_math_and_design/M6/THEOREM_M6.md:66:## 2. Tier 0 • analytic rejection before any host work
research/stage4_math_and_design/M6/THEOREM_M6.md:68:### 2.1 Bayes-risk brackets for each evidence state
research/stage4_math_and_design/M6/THEOREM_M6.md:80:### 2.2 Exact group action and reference algebra
research/stage4_math_and_design/M6/THEOREM_M6.md:88:## 3. Tier 1 • CPU geometry and the frozen host
research/stage4_math_and_design/M6/THEOREM_M6.md:94:### 3.1 Rank, curvature and optimization certificates
research/stage4_math_and_design/M6/THEOREM_M6.md:102:### 3.2 Geometry of $W_{\rm out}$ and vocabulary readout
research/stage4_math_and_design/M6/THEOREM_M6.md:116:## 4. Tier 2 • sequential decisions and an exact cost envelope
research/stage4_math_and_design/M6/THEOREM_M6.md:118:### 4.1 VoI and continuation
research/stage4_math_and_design/M6/THEOREM_M6.md:128:### 4.2 Two-look design and what the Monte Carlo verifies
research/stage4_math_and_design/M6/THEOREM_M6.md:154:### 4.3 Stage 4 Modal plan in US dollars
research/stage4_math_and_design/M6/THEOREM_M6.md:178:## 5. Stage 4 → 6 critical path
research/stage4_math_and_design/M6/THEOREM_M6.md:215:## Replay
````

### Event #31: `item.completed` — `command_execution` (`item_27`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$env:UV_CACHE_DIR=(Resolve-Path '"'research/stage4_math_and_design/M4/.uv-m4-r2').Path; "'$env:TEMP=$env:UV_CACHE_DIR; $env:TMP=$env:UV_CACHE_DIR; $env:UV_OFFLINE='"'1'; "'$env:UV_NO_PROJECT='"'1'; "'$env:UV_LINK_MODE='"'copy'; "'$env:OPENBLAS_NUM_THREADS='"'1'; uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B research/stage4_math_and_design/M4/verify_m4.py"
```

**Aggregated Output:**

```text
Installed 2 packages in 7.23s
Verified frozen s17 identity and pinned Python/NumPy/SciPy
Verified static splits, historical manifests and Modal arithmetic
PASS 168 mapping cells, 32 nuisance cells, 6 dual-cache profiles, 219 causal bundles, results_m4.json
```

### Event #32: `item.completed` — `command_execution` (`item_29`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n -A 10 -B 2 'eval_counts =|schedule =|probe.*synthetic|8\\.696' research/stage4_math_and_design/M4/verify_m4.py"
```

**Aggregated Output:**

```text
626-    rng = np.random.default_rng(SEED + 3)
627-    preconditions = ('train', 'dev', 'test')
628:    schedule = [('pluto', f'pluto/{name}', name, 20, 13, (15, 18), (21, 26), (34, 42, 160))
629-                for name in preconditions]
630-    schedule += [('cttc', f'{a["capture_id"]}/{b}/{name}', name, 4, 6, (126, 127), (40, 48), (140, 148, 500140))
631-                 for a in anchors for b in range(8) for name in preconditions]
632-    synth_ids = set()
633-    bad_checks, prefix_checks, max_gauge = 0, 0, 0.0
634-    for dataset, unit, split, sps, n_sync, ref_slots, holdout, times in schedule:
635-        cutoff = sps * times[1] + 3
636-        sync = [synthetic_acquisition(unit, split, k, sps, rng) for k in range(n_sync)]
637-        refs = [synthetic_acquisition(unit, split, k, sps, rng) for k in ref_slots]
638:        probes = [synthetic_acquisition(unit, split, k, sps, rng) for k in range(*holdout)]
639-        windows = [synthetic_acquisition(unit, split, k, sps, rng) for k in times]
640-        for row in sync + refs + probes + windows:
641-            assert row.acquisition_id not in synth_ids
642-            synth_ids.add(row.acquisition_id)
643-        f0 = {row.acquisition_id: row for row in sync + windows[:2]}
644-        allowed = {row.acquisition_id: row for row in refs}
645-        assert not set(allowed) & set(f0)
646-        assert max(r.raw_support[1] for r in refs + sync) <= cutoff < windows[-1].raw_support[0]
647-        covariance = np.array([[.5, .17 - .04j], [.17 + .04j, .6]])
648-        ancillary = np.sqrt(V0) * noise(rng, (2, 2))
--
907-    stress = splits['pluto']['stress_rows'] + splits['cttc']['stress_rows']
908-    synthetic = 32 * 64
909:    eval_counts = {'pluto_raw': len(arms) * (pluto['eval_counts']['test'] * 4 + pluto['repeat_candidate_rows'] * 2),
910-                   'cttc_raw': len(arms) * (cttc['counts']['test_target'] * 4 + cttc['repeat_candidate_rows'] * 2),
911-                   'real_stress_q2': len(arms) * stress, 'synthetic_blind': len(arms) * synthetic * 5,
912-                   'dev_calibration': len(arms) * 4000}
913:    assert eval_counts == {'pluto_raw': 53760, 'cttc_raw': 28800, 'real_stress_q2': 8240,
914-                           'synthetic_blind': 40960, 'dev_calibration': 16000}
915-    pilot_eval_rows = 2048
916-    pilot_seconds = math.ceil(pilot_steps * Decimal('2.50')) + math.ceil(pilot_eval_rows * Decimal('0.030')) + 155
917-    assert pilot_seconds == 650
918-    continuation_steps = len(arms) * steps - pilot_steps
919-    training_seconds = math.ceil(continuation_steps * Decimal('2.50'))
920-    eval_seconds = {key: math.ceil(value * Decimal('0.030')) for key, value in eval_counts.items()}
921-    tasks = [('tier2_futility_pilot', 'A100-80GB', pilot_seconds, 2, 16),
922-             ('continuation_if_go', 'A100-80GB', training_seconds, 2, 16)]
923-    tasks += [(key, 'A100-80GB', seconds, 2, 16) for key, seconds in eval_seconds.items()]
--
941-    assert total == gpu_seconds * rate['A100-80GB'] + cpu * sum(row[2] * row[3] for row in tasks) + mem * sum(row[2] * row[4] for row in tasks)
942-    assert pilot_seconds * a100 == Decimal('0.49121800')
943:    assert gpu_seconds == 10492 and total == Decimal('8.69671424') and total < target < free - past
944-    holdout_rows = len(arms) * (160 * 4 + 80 * 2)
945-    assert holdout_rows == 3200 and eval_counts['cttc_raw'] == holdout_rows * 9
946-    extra_pair = math.ceil(2 * 2 * steps * Decimal('2.50')) * a100
947-    assert total + extra_pair > target  # excludes optional additional scoring, so already infeasible
948-    # Engineering/futility only. This checks the planning reference, not a live dev outcome.
949-    info = 0.18
950-    posterior_variance = 1 / (.25 + info)
951-    prediction_sd = np.sqrt(1 - info + (1 - info) ** 2 * posterior_variance)
952-    slope = np.sqrt(info) * (1 + (1 - info) * posterior_variance)
953-    # Verify M6's reference cut by evaluating the exact Gaussian tail with its rounded z and boundary.
```

### Event #33: `item.completed` — `file_change` (`item_30`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M4\\verify_m4.py",
    "kind": "update"
  }
]
```

### Event #34: `item.completed` — `file_change` (`item_31`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M4\\THEOREM_M4.md",
    "kind": "update"
  }
]
```

### Event #35: `item.completed` — `command_execution` (`item_32`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$r=Get-Content -Raw research/stage4_math_and_design/M4/results_m4.json | ConvertFrom-Json; foreach ($c in $r.dual_cache_ceiling_surrogates.cells) { '"'{0} {1}: {2:F5} {3:F5} {4:F5}; {5:F5} {6:F5} {7:F5}; {8:F5}' -f "'$c.profile,$c.snr_db,$c.nll_nats.s17.symbol,$c.nll_nats.full16.symbol,$c.nll_nats.continuous_bayes.symbol,$c.nll_nats.s17.joint,$c.nll_nats.full16.joint,$c.nll_nats.continuous_bayes.joint,$c.paired_mc_standard_error_nats.joint }; $r.dual_cache_ceiling_surrogates.max_polar_quadrature_l1_order_10_vs_14; $r.causal_frontend_contract | Select-Object probability_gauge_max_abs_defect,schur_v_joint_abs_log_likelihood_error,packed_bytes_cap | ConvertTo-Json -Compress'
```

**Aggregated Output:**

```text
Pluto_fade -2: 1.15279 1.11597 1.10286; 1.80240 1.47241 1.39833; 0.00404
Pluto_fade 3: 0.81171 0.75042 0.73408; 1.43449 1.10624 1.03194; 0.00406
Pluto_fade 8: 0.34509 0.29459 0.28550; 0.96959 0.64325 0.57740; 0.00377
CTTC_spur -3: 1.12425 1.07443 1.06097; 1.77413 1.42613 1.35206; 0.00409
CTTC_spur 10: 0.11027 0.07219 0.07022; 0.75140 0.42443 0.36524; 0.00356
CTTC_spur 20: 0.00007 0.00000 0.00000; 0.67244 0.37820 0.29676; 0.00320
7.64993458048926E-16
{"probability_gauge_max_abs_defect":7.7715611723760958E-16,"schur_v_joint_abs_log_likelihood_error":8.8817841970012523E-16,"packed_bytes_cap":87768}
```

### Event #36: `item.completed` — `file_change` (`item_33`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M4\\THEOREM_M4.md",
    "kind": "update"
  }
]
```

### Event #37: `item.completed` — `file_change` (`item_34`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M4\\verify_m4.py",
    "kind": "update"
  }
]
```

### Event #38: `item.completed` — `file_change` (`item_36`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M4\\verify_m4.py",
    "kind": "update"
  }
]
```

### Event #39: `item.completed` — `command_execution` (`item_35`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$env:UV_CACHE_DIR=(Resolve-Path '"'research/stage4_math_and_design/M4/.uv-m4-r2').Path; "'$env:TEMP=$env:UV_CACHE_DIR; $env:TMP=$env:UV_CACHE_DIR; $env:UV_OFFLINE='"'1'; "'$env:UV_NO_PROJECT='"'1'; "'$env:UV_LINK_MODE='"'copy'; "'$env:OPENBLAS_NUM_THREADS='"'1'; uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B research/stage4_math_and_design/M4/verify_m4.py --check-http"
```

**Aggregated Output:**

```text
Verified frozen s17 identity and pinned Python/NumPy/SciPy
Verified static splits, historical manifests and Modal arithmetic
PASS 168 mapping cells, 32 nuisance cells, 6 dual-cache profiles, 219 causal bundles, results_m4.json
```

### Event #40: `item.completed` — `file_change` (`item_37`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M4\\THEOREM_M4.md",
    "kind": "update"
  }
]
```

### Event #41: `item.completed` — `agent_message` (`item_39`)

The dual-cache test shows a 0.294–0.348 nat reduction in joint symbol-and-variance NLL across the six synthetic profiles. At +20 dB, almost all of that gain comes from identifying the variance state. I’ve also priced the fixed-text interventions separately, so they fit inside the same $9.50 cap.

### Event #42: `item.completed` — `command_execution` (`item_38`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$env:UV_CACHE_DIR=(Resolve-Path '"'research/stage4_math_and_design/M4/.uv-m4-r2').Path; "'$env:TEMP=$env:UV_CACHE_DIR; $env:TMP=$env:UV_CACHE_DIR; $env:UV_OFFLINE='"'1'; "'$env:UV_NO_PROJECT='"'1'; "'$env:UV_LINK_MODE='"'copy'; "'$env:OPENBLAS_NUM_THREADS='"'1'; uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B research/stage4_math_and_design/M4/verify_m4.py --check-http"
```

**Aggregated Output:**

```text
Verified frozen s17 identity and pinned Python/NumPy/SciPy
Verified static splits, historical manifests and Modal arithmetic
PASS 168 mapping cells, 32 nuisance cells, 6 dual-cache profiles, 219 causal bundles, results_m4.json
```

### Event #43: `item.completed` — `file_change` (`item_40`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M4\\reply_r2.md",
    "kind": "add"
  }
]
```

### Event #44: `item.completed` — `command_execution` (`item_41`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n '8\\.44|8\\.84|9\\.50|2\\.901|Round 1|Round 2|pilot.*0\\.491|PhaseContext' research/stage4_math_and_design/M4/THEOREM_M4.md research/stage4_math_and_design/M4/reply_r2.md; git rev-parse HEAD; git --no-optional-locks status --short --untracked-files=all -- research/stage4_math_and_design/M4"
```

**Aggregated Output:**

```text
research/stage4_math_and_design/M4/reply_r2.md:1:# M4 Round 2 · Stage 4 handoff
research/stage4_math_and_design/M4/reply_r2.md:3:**The four-arm A100 plan schedules $8.84181248 under a $9.50 gross cap and requires a $0.00 net spend guard.** Standard A100-80GB Function pricing is `$0.000694 + 2×$0.0000131 + 16×$0.00000222 = $0.00075572/s`. The Round 1 host allocation was 12 GiB, which explains the `$0.00000888/s` difference. Stage 3D observed `52,972.5 MB` training and `45,409.9 MB` eval GPU peaks. The default ledger assigns all GPU rows to A100-80GB and requires a fresh Stage 4 throughput and parity receipt. A 24 GiB alternative needs a separately charged ≤$0.10 receipt proving ≤20 GiB peak and scorer parity before any blind rows. [Modal pricing](https://modal.com/pricing) · [Stage 3D measurements](../../stage3d_qwen_signed_phase/REPORT.md)
research/stage4_math_and_design/M4/reply_r2.md:5:- **Pilot and scope.** Buy up to **173 Host-Pure steps and 2,048 disjoint dev scores for $0.49121800**. Continue only after engineering and predictive futility gates pass. M6’s 0.18 information fraction is a planning-model parameter. The Stage 4 arm has 384 steps, of which the pilot already pays for 173. On go, spend 1,363 more steps across the four matched arms for **$2.57549376**. Freeze all four at seed `202611`. Inference is conditional on those connectors. Two extra seeds for each coupled arm cost **$2.90196480** in training alone and sit outside this plan.
research/stage4_math_and_design/M4/reply_r2.md:8:- **Causality and actions.** `PhaseContext` carries signed `omega_hat`, shared `v_omega`, diffusion and calibrated SNR. Reveal only Pluto `15,18` or CTTC `126,127`. Stable acquisition IDs exclude evidence already in `F0`; use joint conditional innovations for correlated observations. A common RX gauge permutes only Q3 and a TX relabel permutes only Q2. The CPU replay exercises byte-identical canonical projections, forbidden reads, conditional likelihood factorization and both physical probability actions across 219 simulated split fixtures. Real raw-support and timing receipts are still required. Preserve all registered rows with a common erasure rule and label-availability accounting. [Causal contract](THEOREM_M4.md#25-conditional-updates-and-group-actions)
research/stage4_math_and_design/M4/THEOREM_M4.md:3:Round 2 lock. All logarithms and information quantities use nats. One causal CPU pass emits both the frozen SNR-preserving `s17` code and the M5 16-bit packet. Stage 4 GPU comparisons retain the `s17` prefix with the same evidence across four frozen connectors. The 16-bit and continuous views enter physical CPU comparisons. The measured radio frontend and GPU continuation have separate qualification gates.
research/stage4_math_and_design/M4/THEOREM_M4.md:165:| **PhaseContext** | `t1,t2,t3 <i8` in fixed ticks, `target_valid u1`, `clock_id` and `time_unit=symbol`, `ticks_per_symbol=2^20`. Convert certified matched clock coordinates to ticks once using nearest ties-to-even. Subtract integer `t1` before converting a lag to float64 symbol units. Sub-symbol timing and actual raw availability are also in the provenance record. Pluto uses local frame clocks with the known repeat spacing. CTTC uses the causal sample-rate and clock map, including SFO. Unknown Pluto inter-frame times get a missing clock and cannot feed signed-phase transport. Synthetic uses the same contract with `0,h,h+ℓ`. |
research/stage4_math_and_design/M4/THEOREM_M4.md:166:| **PhaseContext nuisance state** | `omega_hat <f8` (alias `omega_rad_per_unit`), `v_omega <f8`, `diffusion_var_per_unit <f8`, `snr_linear <f8[2]`, `estimate_tick <i8`, `estimate_age_ticks <i8`, `phase_anchor_tick <i8`, `anchor_mean/anchor_cov` for allowed gauge-dependent state, `phase_model_id`, `state_digest`, `valid` flags. Sign is always \(T(t)\propto e^{+i\widehat\omega(t-t_1)}\). If deterministic CFO is removed in the frontend, `omega_hat` is the residual in that exact coordinate system. SNR is matched signal energy divided by proper-complex variance. Keep phase/rate correlation or weighted joint hypotheses when fitted. An opaque state digest is audit metadata. |
research/stage4_math_and_design/M4/THEOREM_M4.md:168:| **RefPool innovation inputs and outputs** | `K_ref <c16[2,2]` (raw complex noise within allowed pilot block), `K_ref_B` and a causal joint law or conditional innovation kernel if `B` is correlated. Align the matrix with canonical reference order. A later single-length view uses its principal submatrix and masks empty slots. Pass the shared rate state once via `PhaseContext`, plus anchor prior and state/ledger identifiers. Publish fixed-shape canonical `q2[4]`, `q3[4]` and optional `joint[4,4]` or phase-node likelihood factors with clocks and normalization. The host projector accepts fixed `uint8` masks and canonical float32 numeric features. Canonicalize signed zero at serialization. |
research/stage4_math_and_design/M4/THEOREM_M4.md:219:3. **Fixed-connector conditional scope.** Complete the remaining 211 Host-Pure steps and 384 steps each for the other three arms only after go. Conditional on dev, selection, host, four checkpoints and calibrators, use a valid blind procedure. For a true null the probability of `go ∧ reject` is \(\mathbb E[1_{\rm go}\Pr(\mathrm{reject}\mid\mathrm{dev})]\le\alpha\) when that blind test is conditionally valid. The 32×64 synthetic blind plan is a budgeted conditional cohort. M2's 131-block design is an independent full-width power option, not funded as a 131-block four-arm five-question rerun. Freeze minimum per-endpoint detectable effects and power on independent dev before release. If 32 blocks or the fixed real cohorts lack precision, report conditional feasibility and interval estimates without promotion. The historical proxy's ten-seed target concerns retraining populations. Here the statistical unit is the blind cluster conditional on the fixed connectors. Two additional seeds for each of the coupled arms alone cost **$2.90196480** for training, plus evaluation, and exceed the cap on top of the scheduled workload. Skip those fits in this locked schedule. [M2 inference and scope](../M2/THEOREM_M2.md)
research/stage4_math_and_design/M4/THEOREM_M4.md:225:Locked standard Functions price formula is `billed_seconds × (GPU + physical_cores×0.0000131 + GiB×0.00000222)`. A100-80GB base is `$0.000694/s` (`$2.4984/h`), A100-40GB `$0.000583/s`, L4 `$0.000222/s`, and A10/A10G `$0.000306/s`. With 2 physical cores and **16 GiB**, A100-80GB is **$0.00075572/s** (`$2.720592/h`), L4 `$0.00028372/s`, and A10 `$0.00036772/s`. M4 Round 1 used 12 GiB and therefore `$0.00074684/s`. The delta `$0.00000888/s` is exactly `4×0.00000222`. M6's 4,000 L4 real-eval seconds would cost an extra **$1.888** at the A100 base GPU rate with the same host sizing. This ledger instead uses 2,477 A100 raw-real seconds at the registered 30 ms/row cap, an A100 benchmark hypothesis to verify on the receipt. M6's earlier 2,707 continuation steps bought three seeds of one candidate. This ledger buys four matched 384-step arms conditional on one seed. [Modal pricing](https://modal.com/pricing) · [Modal resource billing and caps](https://modal.com/docs/guide/resources)
research/stage4_math_and_design/M4/THEOREM_M4.md:229:| A100 Tier-2 pilot, `433+62+155` | 650 | 0.49121800 |
research/stage4_math_and_design/M4/THEOREM_M4.md:242:| **Scheduled**, 10,684 A100 seconds and 3 CPU function hours | | **8.84181248** |
research/stage4_math_and_design/M4/THEOREM_M4.md:244:| **Hard gross Stage 4 cap** | | **9.50000000** |
research/stage4_math_and_design/M4/THEOREM_M4.md:246:CPU-only jobs total `$0.76770000`. The scheduled plan leaves `$4.03818752` against the historical `$12.88`, and a full `$9.50` authorization leaves `$3.38`. Budget cold starts, stopped shards, image preparation, Volume commit/reload, teardown and retries within the priced ledger. Use one bounded billed invocation at a time, disable automatic retries/fanout, precharge the worst-case next shard before dispatch, and stop if the remaining scope or allowance is insufficient. Commit checkpoint and input hashes to Resume only on a recorded go. Verify live remaining compute and aggregate storage/egress quota. The arithmetic is exact under these caps. Completion depends on actual causal alignment and measured throughput.
research/stage4_math_and_design/M4/THEOREM_M4.md:248:Require account-visible `$0.00` out-of-pocket gating and sufficient remaining compute and volume/egress quotas before launch. On Modal Starter a workspace spend limit caps net charges separately from the workspace gross usage limit. Verify the owner-set net `$0.00` limit or equivalent hard guard, the live compute balance and the gross usage cap. Use dashboard and control-plane caps alongside the serial local ledger because other usage and reporting lag affect the available balance. A frozen self-hosted model on Functions keeps the scheduled workload in compute charges. Round 2 performed zero cloud dispatches and did not inspect live workspace balances. [Modal budgets](https://modal.com/docs/guide/budgets) · [Modal billing](https://modal.com/docs/guide/billing)
research/stage4_math_and_design/M4/THEOREM_M4.md:267:### 4.1 Round 2 dual-cache ceiling replay
387e7bc749150a69fb317b3f1351ad0d07b3bfad
?? research/stage4_math_and_design/M4/THEOREM_M4.md
?? research/stage4_math_and_design/M4/reply_r1.md
?? research/stage4_math_and_design/M4/reply_r2.md
?? research/stage4_math_and_design/M4/results_m4.json
?? research/stage4_math_and_design/M4/verify_m4.py
```

### Event #45: `item.completed` — `command_execution` (`item_42`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$r=Get-Content -Encoding utf8 -Raw research/stage4_math_and_design/M4/results_m4.json | ConvertFrom-Json; $r | Select-Object round,status,python,numpy,scipy | ConvertTo-Json -Compress; $r.causal_frontend_contract | Select-Object fixture_bundles,byte_exact_refpool_fixture_bundles,rejected_forbidden_ref_attempts,probability_gauge_max_abs_defect,schur_v_joint_abs_log_likelihood_error,packed_bytes_cap,packed_bytes_with_optional_separate_nibbles_cap | ConvertTo-Json -Compress; $r.modal_budget | Select-Object stage4_scheduled_usd_cap,stage4_total_usd_cap,unallocated_reserve_below_hard_cap_usd,stage4_a100_seconds_cap,extra_two_seeds_for_two_arms_training_only_usd,headroom_to_historical_remaining_usd | ConvertTo-Json -Compress; $r.dataset_manifests.live_http | ConvertTo-Json -Compress'
```

**Aggregated Output:**

```text
{"round":2,"status":"pass","python":"3.12.7","numpy":"2.3.5","scipy":"1.17.0"}
{"fixture_bundles":219,"byte_exact_refpool_fixture_bundles":219,"rejected_forbidden_ref_attempts":1533,"probability_gauge_max_abs_defect":6.6613381477509392E-16,"schur_v_joint_abs_log_likelihood_error":0.0,"packed_bytes_cap":87768,"packed_bytes_with_optional_separate_nibbles_cap":109710}
{"stage4_scheduled_usd_cap":"8.84181248","stage4_total_usd_cap":"9.50","unallocated_reserve_below_hard_cap_usd":"0.65818752","stage4_a100_seconds_cap":10684,"extra_two_seeds_for_two_arms_training_only_usd":"2.90196480","headroom_to_historical_remaining_usd":"4.03818752"}
{"mendeley_files":{"metadata_reachable":false,"detail":"HTTP Error 403: Forbidden"},"zenodo_files":{"http_status":200,"metadata_reachable":true}}
```

### Event #46: `item.completed` — `command_execution` (`item_43`)

- Status: `failed`, Exit Code: `1`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg --files research/stage4_math_and_design/M1 research/stage4_math_and_design/M2 research/stage4_math_and_design/M3 research/stage4_math_and_design/M5 research/stage4_math_and_design/M6 -g 'reply_r2.md'; rg -n 'm4.real_dual_cache|8\\.84181248|Tier-2|r_rx|tx_anchor' research/stage4_math_and_design/M1/reply_r2.md research/stage4_math_and_design/M3/reply_r2.md research/stage4_math_and_design/M6/reply_r2.md"
```

**Aggregated Output:**

```text
research/stage4_math_and_design/M2\reply_r2.md
research/stage4_math_and_design/M6\reply_r2.md
research/stage4_math_and_design/M3\reply_r2.md
rg: research/stage4_math_and_design/M1/reply_r2.md: 系统找不到指定的文件。 (os error 2)
```

### Event #47: `item.completed` — `command_execution` (`item_46`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command 'Get-Content -Encoding utf8 research/stage4_math_and_design/M2/reply_r2.md -TotalCount 45'
```

**Aggregated Output:**

```text
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
- **M3** · Require the paired Q2 NLL ordering two-valid \(>\) one-valid \(>\) none in utility, and exact 0.0 output change for copies, distractors, padding and list order. Check the actual scorer path. Blind temperature fits stay retrospective.
- **M4/M6** · Use IUT within each of the three claim slots and Holm at 0.05 across synthetic+Q2, Pluto and CTTC. Plan CTTC with **nine capture units** when their conditional independence is defensible. The f708_snr10 result is a held-out cohort guard. Hold the capture out of every fit and calibration. Shared cyclic TX source restricts generalization.
- **M5** · Require Stage 6 max EIG to beat random, nearest and dev-frozen fixed-best on both post-reveal NLL and accuracy at the same one-read budget. Pair the 16-bit canonical vs 4-bit S17 representation test separately under the same prescribed action.

The pinned CPU verifier exited 0. The Stage 3D replay still has maximum in-scope summary difference zero. [Gate and proof](THEOREM_M2.md) · [Verifier](verify_m2.py) · [Numerical results](results_m2.json)

```

### Event #48: `item.completed` — `command_execution` (`item_44`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command 'Get-Content -Encoding utf8 research/stage4_math_and_design/M6/reply_r2.md -TotalCount 45'
```

**Aggregated Output:**

```text
# M6 Round 2 • Stage 4–6 execution handoff

**Use one A100-80GB ledger and one fixed training seed.** Run four matched initial fits of **69/384 steps (17.96875%)** for Zero, Naive-Additive, Host-Pure and Physical+HostResidual. A dev-only futility pass authorizes the remaining 315 steps of each fit. The final estimates condition on these checkpoints and the registered physical units. Stage 3D measured 52,972.5 MiB in training and 45,409.9 MiB in evaluation; the new longest-case A100 receipt must pass **65,000 MiB, 2.50 s/step, 30 ms/row**, and **\(10^{-5}\)** probability parity.

| Stage 4 debit | Maximum USD |
|---|---:|
| CPU ingest, causal frontend, dual cache, readers and audits | 0.76770000 |
| A100 setup, 69-step pilot per arm and disjoint pilot-dev scoring | 0.87663520 |
| Post-GO four-arm completion, final dev, synthetic core and remaining setup | 4.14814708 |
| Pluto natural, 2,240 frames / 16 blocks | 1.21897636 |
| CTTC natural, 1,440 packets / nine sessions, including held-out `f708_snr10` | 0.65294208 |
| Real stress and explicit slot/text controls | 0.72624692 |
| Inside-cap reserve (including the controlled retry line) | 1.10935236 |
| **Hard gross cap / external buffer versus historical \$12.88** | **9.50 / 3.38** |

The total before retry is **\$8.39064764**; the explicit retry line raises scheduled max to **\$9.07079564**. All CPU plus pilot exposure before GO is **\$1.64433520**, before any separately charged reserve. Live credits, resource ceilings and the **\$0.00** net-spend guard need receipts before dispatch. [Modal pricing](https://modal.com/pricing), [Modal budgets](https://modal.com/docs/guide/budgets)

- **Tier 0:** Build both cache views in one causal CPU pass. The 16-bit word recovers s17 by right shift. Verify all cutoffs, source guards, exact cache prefixes and sentinel hashes. RefSufficientPool charges each acquisition once; correlated fresh observations contribute a conditional innovation. Exclude the noncausal whole-capture and full-source aligners.
- **Tier 1:** Qualify **both** Host-Pure and the hybrid's actual residual Jacobian, with held-out CFO/angle tests and matched inputs. Arm 1 answers from Qwen scores alone. Compare Arm 2 with its residual-zero physical prior and fixed-prior shuffled residual. Fixed-position slot/text interventions identify the two evidence paths within the checkpoint.
- **Tier 2:** Spend zero efficacy alpha on the pilot. Use separate calibration dev and blind cohorts. Apply the domain intersection-union gates and Holm at .05. CTTC's named holdout is one capture, so scope its transfer result accordingly.
- **Stages 5–6:** Carry the same versioned cache, pool and canonical state into event and active-read tests. A shared rank-16 affine adapter plus temperature needs **43,521 / 69,633 / 87,041** parameters on 4B/9B/27B. Each actual host still needs a geometry and memory receipt. Seal the LEMMO case intersection and evidence rights before scoring. Report native comparisons when a cache-matched interface is unavailable.

**Verified locally.** The pinned uv run passed all 102 Stage 3D accuracy/NLL parity checks, maximum discrepancy **3.56×10⁻¹⁶**. It verified the ledger, 65,536 packet layouts and the algebraic causality, rotation, innovation and adapter fixtures. The Round 2 **10,000-trial** model gave **59.9839%** simulated savings versus the same-scope fixed ledger (**59.8771%** by integration), **94.040%** working single-endpoint power and **0.7333%** boundary-null promotion at a **1.6667%** first-Holm threshold. Real joint power, raw-data causality, the actual-host Jacobians, provider cap and A100 throughput remain unverified. No cloud job ran. [Full protocol and DAG](THEOREM_M6.md) · [Verifier](verify_m6.py) · [Results](results_m6.json)
```

### Event #49: `item.completed` — `command_execution` (`item_45`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command 'Get-Content -Encoding utf8 research/stage4_math_and_design/M3/reply_r2.md -TotalCount 45'
```

**Aggregated Output:**

```text
# M3 Round 2 | Correlated pilots and Stage 6 forecasts

**The causal ledger and noise boundary serve different purposes.** `RefSufficientPool` applies the Schur-complement likelihood only to fresh, unique IDs. An old ID adds exactly zero evidence, even if its observation enters the conditional mean of a new correlated pilot. If the old observation is needed for that mean, the adapter must retain its value or supply an equivalent conditional predictive law. The scalar Gaussian witness shows the cost of treating a copy as independent. The reported variance falls from **0.303030 to 0.172414**, with **0.298134 nats** of expected forward-KL regret. [Derivation and v2 schema](THEOREM_M3.md#7-theorem-2-extension--conditioning-on-causal-history) · [Implementation](ref_pool.py)

**CPU checks passed.** The conditional likelihood agrees with the full Gaussian block to \(8.0\times10^{-15}\). The v2 adapter agrees with `q2_reader` to \(8.9\times10^{-16}\) across all 512 blind C4 mothers. Duplicate, distractor, padding and order tests give byte-identical `uint8` headers, `float32` evidence, and deterministic prefix embeddings, IDs, positions and masks. Maximum KL and TV are both **0.0**. The Pluto/CTTC tests exercise the registered role and clock layouts with synthetic observations. [Verification receipt](results_m3.json)

**Shared CFO matters.** On 4,096 paired synthetic draws at clocks \(1,7\), with conditional thermal correlation magnitude `0.416`, target \(G\), diffusion variance `0.0484` per unit and CFO-rate variance `0.024`, the full two-clock model gives **0.147654 NLL**.

| Alternative | NLL | Paired excess over full |
|---|---:|---:|
| Full observation with Brownian-only phase | 0.236292 | +0.088638 |
| Correct posterior given the derotated sum | 0.378354 | +0.230700 |
| Coherent single-sum approximation | 0.593499 | +0.445845 |

**Stage 6 uses the same likelihood before and after reveal.** The fixture gives **0.433721 nats** EIG to the selected probe, versus numerical zero for the aliased candidate. A single read moves the jump posterior from **0.45 to 0.92395**. The pre-reveal log marginal and posterior match the corresponding post-reveal values to floating-point precision. Probe time `7` follows target time `4`. The fixture permits retrospective inference at its declared deadline and rejects the early reveal. The deployment vault owns the one-read budget. [Predictive contract](THEOREM_M3.md#10-stage-6-pre-reveal-mixture-and-post-reveal-update)

**Dev-locked C4 contrast.** The residual uses fixed cap `0.15` on centered, temperature-calibrated host scores. The native columns come from archived Stage 3D Qwen logits and the residual is applied offline. This checks the score path on matched mothers. Actual Round 2 Qwen forwards remain an integration test.

| Q2 arm | Native Qwen NLL | Physical + offline residual NLL | `q2_reader` NLL |
|---|---:|---:|---:|
| One valid pilot | 1.388883 | 0.819535 | 0.819207 |
| Two fresh pilots | 1.405640 | 0.664052 | 0.663189 |

The measured residual increments are **+0.000328 ± 0.000608** and **+0.000863 ± 0.000598** nats, with paired block SEs. This comparison provides no evidence of a residual gain.

**Integration handoff**

- **M1**. Consume the two fixed canonical slots and keep the binary64 joint posterior on the sidecar. Apply receiver-frame transport to Q3 RX sectors outside the host.
- **M4**. Include every consumed sync or history ID in \(\mathcal F^0\). Pass the calibrated observation law and raw-support availability for Pluto `15,18` and CTTC `126,127`. Estimate conditional \(K_{\rm ref}\) separately from latent CFO and leave retrospective probes out of forward Q3.
- **M5**. Commit the hypothesis and marginal mixtures, chosen ID, and deadline before release. Use the same innovation density for the single selected update.
- **M2/M6**. Gate the exact wire invariances and compare the native and residual models on the same dev-locked forecasts and blind tuples. Retain the causal slot-swap control before attributing a gain to frozen-host inference.

The default CPU verifier exited **0** via `uv run --offline --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 --with torch python -B research/stage4_math_and_design/M3/verify_m3.py` in **32.28 s** wall time. It preserves the slow Round 1 simulation results and recomputes the C4 archive replay and Round 2 checks. [CPU verifier](verify_m3.py) · [Results](results_m3.json)
```

### Event #50: `item.completed` — `command_execution` (`item_47`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n '69|8\\.39064764|9\\.07079564|def .*budget|slot_text|2.50|650|173|all.*A100' research/stage4_math_and_design/M6/THEOREM_M6.md research/stage4_math_and_design/M6/verify_m6.py; rg -n 'class .*|def.*canonical|timestamp|omega_hat' research/stage4_math_and_design/M3/ref_pool.py | Select-Object -First 30; rg -n '02 nats|2 pp|capture' research/stage4_math_and_design/M2/THEOREM_M2.md | Select-Object -Last 12"
```

**Aggregated Output:**

```text
research/stage4_math_and_design/M6/verify_m6.py:37:    "pilot_steps_per_arm": 69,
research/stage4_math_and_design/M6/verify_m6.py:45:    "max_peak_vram_mib": 65000,
research/stage4_math_and_design/M6/verify_m6.py:376:def budget():
research/stage4_math_and_design/M6/verify_m6.py:378:    cpu_rate, ram_rate, gpu_rate = map(d, ("0.0000131", "0.00000222", "0.000694"))
research/stage4_math_and_design/M6/verify_m6.py:408:        ("blind_matched_slot_text_controls", gpu, math.ceil(intervention_rows * row_time), 2, 16, None, intervention_rows, "controls"),
research/stage4_math_and_design/M6/verify_m6.py:432:    assert a100_seconds == 10987 and reserved == d("9.07079564") and unassigned == d("0.42920436")
research/stage4_math_and_design/M6/verify_m6.py:433:    assert (planned, totals["cpu"], totals["pilot"]) == (d("8.39064764"), d("0.7677"), d("0.87663520"))
research/stage4_math_and_design/M6/verify_m6.py:553:    y = np.array([1.15 + .37j, -.41 + .69j])
research/stage4_math_and_design/M6/THEOREM_M6.md:1:# M6 Round 2 • A CPU-gated, all-A100 Stage 4 execution lock
research/stage4_math_and_design/M6/THEOREM_M6.md:3:**Decision.** Reuse the Stage 3D score archive for diagnosis, build both cache views on CPU, and allow at most a disjoint-dev 69-step pilot per arm before any full four-arm fit. All Stage 4 GPU work uses A100-80GB within a **\$9.50** gross cap, leaving **\$3.38** against the historical \$12.88. §§1–3 retain the Round 1 mathematical foundations. §§4–6 state the operative Round 2 schedule and gates. Round 2 used local CPU only.
research/stage4_math_and_design/M6/THEOREM_M6.md:48:| H_c4_035 | 1,024 | 63.867% / .942628 | 63.770% / .950133 | 25.765% / 1.416987 | 23.172% / 1.435077 | .007505 | .466854 | .484944 |
research/stage4_math_and_design/M6/THEOREM_M6.md:50:For G1 the paired 64-mother block 95% intervals are **[.009178, .014998]** for s17 minus Full IQ and **[.491294, .516912]** for phase minus s17. Phase's observable telescope is .012087838 + .478025740 + .026076989 = .516190567 nats, using unit-temperature host scores for the middle contrast. About **97.66%** of phase's excess over Full IQ lies after the s17 readout when measured at locked temperature. The G1 phase-versus-zero uplift did meet Stage 3D's relative target, even though both arms exceeded the log-loss of a uniform predictor (1.386294). G1's co-rotation worst coordinate defect was **.380019** in phase and near machine precision for the direct readout. Offline 16-way group averaging brought phase NLL to **1.411312** and accuracy to **17.050%**. On the archive's sampled rotations these are distinct endpoints. The all-rotation `H_c4_035` row above is a fresh replay from the archive. Its identity-rotation, both-sign host metrics match Work F. The separate positive-sign Stage 3B lookup anchor in the master JSON is a different comparison.
research/stage4_math_and_design/M6/THEOREM_M6.md:55:| Valid | 66.406% / .819207 | 28.776% / 1.388883 | 26.628% / 1.410690 | .569676 | .591483 |
research/stage4_math_and_design/M6/THEOREM_M6.md:56:| Duplicate, dedup | 66.406% / .819207 | 28.776% / 1.388883 | 26.628% / 1.410690 | .569676 | .591483 |
research/stage4_math_and_design/M6/THEOREM_M6.md:163:Buy exactly **69 of 384 steps per arm (17.96875%)**, then 1,000 pilot-dev rows per arm. Training, pilot dev, post-fit calibration dev, and blind data are disjoint by mother and eligible physical/source support. Exclude `f708_snr10` from fit, shared-prior tuning, pilot and calibration. Preserve checkpoint, optimizer, RNG and data position for resume. The pilot spends **zero efficacy alpha**. Continue only when the registered independent dev model gives predictive probability $\ge.10$ of clearing the synthetic joint gate, the engineering predicates pass, and powered enrollment is justified. Each real branch also requires its own predictive probability $\ge.10$ before its blind rows are funded. Calibrate the map from 69 steps to final effects on disjoint development evidence. A step fraction supplies no statistical information fraction. When the law or enrollment is unresolved take HOLD for the confirmatory claim. An unfunded branch contributes $p=1$ to the family.
research/stage4_math_and_design/M6/THEOREM_M6.md:180:The 10,000-candidate replay shows mean scheduled cost **\$3.357607 vs \$8.390648** (**59.9839%** simulated saving, **59.8771%** integrated), **94.040%** working endpoint power (**94.1785%** integrated), and **0.7333%** boundary-null promotion (**0.7220%** integrated), under the modeled mixture, effect sizes and dev slope. CPU costs are charged even on screen rejection. The fixed single-endpoint boundary alpha is 1.6667%. Round 1's separate parametric O'Brien–Fleming sensitivity replay still returns 64.3619% cost saving and 4.4695% binding Type-I error. Production joint power and the pilot mapping require independent development receipts. [Results](results_m6.json)
research/stage4_math_and_design/M6/THEOREM_M6.md:184:Use **A100-80GB for all GPU work**, as preemptible Functions, at **\$0.00075572/s** including 2 physical cores and 16 GiB. Use hard CPU/memory request-limit pairs and no disk requests that raise the charged memory amount. Modal bills the greater of requested and used resources. M4's **\$8.44073416** includes 10,274 A100 seconds at **\$0.00074684/s** with 12 GiB, plus \$0.76770000 CPU-only preparation. Its 131,760 blind rows exclude 16,000 dev rows. Moving to 16 GiB adds **\$0.09123312** on the same seconds. Round 2 adds 713 A100 seconds for fixed-position counterfactuals. Stage 3D measured **52,972.5 MiB** peak allocated memory in training (microbatch 16) and **45,409.9 MiB** in eval (batch 64). Its source divides bytes by $1024^2$ even though the bench JSON says MB. Use the same hardware class; qualify the new longest cases before continuing. [M4 ledger](../M4/THEOREM_M4.md#33-budget-arithmetic-and-zero-bill-gate), [Stage 3D bench](../../stage3d_qwen_signed_phase/results/bench_steady.json), [Modal rates](https://modal.com/pricing), [Resource controls](https://modal.com/docs/guide/resources)
research/stage4_math_and_design/M6/THEOREM_M6.md:192:| A100 pilot train | 4 × 69 steps | 690 | 0.52144680 |
research/stage4_math_and_design/M6/THEOREM_M6.md:202:| **Scheduled without retries** | **10,087 A100 and 10,800 CPU-only seconds** | | **8.39064764** |
research/stage4_math_and_design/M6/THEOREM_M6.md:204:| **Scheduled with retry** | **10,987 A100 and 10,800 CPU-only seconds** | | **9.07079564** |
research/stage4_math_and_design/M6/THEOREM_M6.md:208:The pilot ceiling is **\$0.87663520**, or **\$1.64433520** including the full conservative CPU allowance. The post-GO arm comparison (remaining fit, calibration, synthetic core and remaining shared overhead) is **\$4.14814708**. Natural Pluto costs **\$1.21897636**, and CTTC **\$0.65294208**, including the named holdout. Stress and interventions add **\$0.72624692**. The reserve inside the cap is **\$1.10935236**, of which \$0.68014800 covers the 900 A100 seconds. The **\$3.38** external credit buffer stays outside the project.
research/stage4_math_and_design/M6/THEOREM_M6.md:214:Require zero OOM and max allocated/reserved/device-used VRAM $\le65000$ MiB on longest registered full gradient, checkpoint-resume and KV scoring workloads, at 512 max prefix tokens and 16 max candidate tokens. Require training-step max elapsed $\le2.50$ s and scoring block time per logical row $\le.030$ s, including padded tails. Use batch 32/microbatch 16 for training, fixed b64 for matching scoring. Hold precision and masks/positions fixed across the compared implementations. Verify maximum probability difference $\le10^{-5}$ on equivalence tests and a same-pattern scorer reference, including expanded linear-attention state. Stage 3D's cross-batch score changes reached .067. Stop for a new parity receipt on any planned batch, precision or kernel change. Do not proceed without the bounded smoke. A100 avoids the known 24 GB mismatch; it still needs a live workload certificate.
research/stage4_math_and_design/M6/THEOREM_M6.md:243:| **4D** | CPU verifies protocol/device/billing before the bounded A100 smoke and 69-step pilot. Require 2.50 s/step, 30 ms/row, 65,000 MiB and $10^{-5}$ probability parity, then dev-only futility and ≥.90 planned joint power for the primary. Commit G1/G3 primary predictions, G2/G4 diagnostics and direct/intervention controls in the scoring vault. |
research/stage4_math_and_design/M6/THEOREM_M6.md:261:For a frozen rank-16 feature projection, use one shared affine host map and one scalar temperature. Parameter count is $(16+1)d_h+1$ even at four slots; fixed slot IDs/positions add no learned offsets. Public text widths $d_h=2560,4096,5120$ yield **43,521 / 69,633 / 87,041** parameters per variant, all below 100,000. This is a shape and capacity bound for the chosen adapter class. If that class fails the candidate geometry test, use a new version to revise it. The CPU verifier constructs orthogonal-frame fixtures with rank-three candidate geometry for both arms. Actual 4B/9B/27B geometry, fit quality and peak memory remain open. The 9B→27B edge changes generation along with size. [4B config](https://huggingface.co/Qwen/Qwen3.5-4B/raw/main/config.json), [9B config](https://huggingface.co/Qwen/Qwen3.5-9B/raw/main/config.json), [27B config](https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json)
26:def canonical_iq(value, frame):
32:def sample_digest(canonical_sample):
38:class PhaseContext:
41:    omega_hat: float
49:        vals = (self.ticks_per_unit, self.omega_hat, self.v_omega, self.diffusion_var_per_unit,
62:        return self.phi0_hat+self.omega_hat*t, cov
66:class ClockMeta:
87:class Observation:
99:class NodeBank:
108:class PoolState:
118:class GaussianLaw:
170:class Forecast:
205:class RefSufficientPool:
249:    def _canonical_records(self, records, as_of, allow_missing=False, allow_outside=False):
358:def fixed_slots(canonical_codes, q2, q3):
365:        raise ValueError("Expected two normalized four-class evidence vectors")
105:3. **Hierarchy and multiplicity.** Keep all repeats, rotations, acquisitions, and matched arms inside their mother and higher cluster. Report a frozen-seed conditional estimate and, if claiming generalization to retraining, a separate random-seed inference. Use equal independent cluster weights for the following exact balanced analysis. Pluto supplies 16 units if they are independent captures. For a CTTC new-session claim use nine independent session means, with all eight clusters per session contributing to that session. A conditional claim on the fixed sessions needs the corresponding estimand and defensible within-session cluster independence. For a permutation or label-randomization gate, lock the allowed exchangeability orbit and recompute the same statistic throughout. Repeated/rotated labels move jointly with their mother and never split correlated session units as though independent. A synthetic randomized design with independent label assignments can instead rerandomize the sanctioned generator operation inside the original design. When there is no justified exchangeability, collect a randomized negative control. State whether it tests a sharp random-label null or a mean-effect margin. A permutation p-value \((1+\#\{T_{\rm perm}\ge T_{\rm obs}\})/(R+1)\) with uniform randomizations and ties included is conservative under the sharp design. Use exact enumeration when feasible.
259:| Pluto proxy | 16 captures | 7 / 5 | .136 / .234 | 15 / 7 | .953 / .994 |
264:At the actual G1 excesses over \(0.02\), the three-fixed-seed G1 or Pluto-proxy minima are 10 and 17 blocks/captures, and the CTTC proxy at ICC .10 gives 4 and 6 sessions. For three random seeds at the planning excesses, the ceiling even with known components and infinitely many blocks is 0.71537 and 0.85941. The conservative finite-level test tends to 0.23839 and 0.30978. These ceilings are consequences of the fitted seed variance and fixed effect-size assumptions. The high potential sampling uncertainty in that variance makes the ten-seed numbers planning proxies. Even with ten seeds, the modeled power statements apply only to the two relative endpoints.
281:- **M4** should keep the 131-block synthetic budget until independent gate-endpoint pilots justify changes. If claiming robustness across retrains, plan around ten or more paired seeds as a G1-based starting point and recalculate using pilot upper variance bounds. With three seeds, scope the main claim to the frozen connectors. Obtain separate Pluto capture and CTTC session variance pilots and freeze session IDs and permutation strata. Preserve all nine CTTC sessions in inference, and use the power objective for the complete gate before locking the real-data sample size.
347:| Calibration | Freeze one dev-fitted scalar per checkpoint/arm and task or predeclared stratum, using the same predeclared search range and objective. Hash the calibrator, priors and adapter with the scores. Reuse the treatment's scalar for its ablations. A synthetic-to-real zero-shot route transfers the synthetic-dev fit and permits only registered causal sample state. A real-dev fit is a separate real-calibrated route and excludes the held-out capture. Choose the confirmatory route on dev. Oracle blind temperature and calibration sensitivity are retrospective diagnostics. |
363:| 3. CTTC + f708_snr10 | Require the same paired natural-data Q2, absolute and exact controls, pilots 126 and 127, and the single-capture transfer guard on f708_snr10. Require \(L_{\rm two}<L_{\rm one}<L_{\rm none}\) and \(L_{\rm two}<L_{\rm prior}\) as fixed-cohort point inequalities there, complete reporting and zero integrity failures. Exclude this capture from checkpoint fitting, calibration and nuisance/reader/prior/hyperparameter tuning. Its causal sync and authorized pilots remain permitted. | 1,440 packets in 72 loops nested in nine captures, and 720 Q3 pairs in 36 disjoint loop-pair groups nested in the same captures. Average paired effects within capture, then weight captures equally. Report the held-out capture separately, plus leave-out results for the other eight. The holdout's p=0/1 bookkeeping is a cohort guard, with no standalone population test. |
367:CTTC replays one public cyclic TX sequence. Disjoint roles/splits prevent direct observation reuse. The common fixed test range \([334096,455000)\) can still induce source-ordinal dependence across captures. Scope the primary cohort estimand to the registered captures and fixed source. Justify independent capture innovations given the locked conditioning before nine-unit inference. If common residual randomness remains across captures and there is no valid design-based alternative, report descriptive results. Show source-ordinal aggregation, capture leave-one-out and capture-by-source sensitivity. The 36/72 nested groups contribute to capture means. They do not supply independent capture or new-source draws. [M4 benchmark and split](../M4/THEOREM_M4.md#2-hardware-benchmark-and-causality-contract)
369:For independent equal-weight Gaussian units with common variance the homogeneous t has \(B-1\) df. In the balanced 32-block synthetic model, fitting eight fixed cell means with common residual variance gives \(32-8=24\) df and the same \(\sigma/\sqrt{32}\) standard error for their average. For unequal allocational counts \(n_h\) with fixed weights \(w_h\), the common-variance standard error is \(\sigma\sqrt{\sum_h w_h^2/n_h}\), with \(B-H\) residual df. For independent Gaussian capture means with fixed heterogeneous means and common variance, the sample variance denominator is noncentral chi-square and independent of the grand mean. It is stochastically larger than the central denominator, giving a conservative upper-tail test at the null boundary. Independent bounded unit means allow the Hoeffding test in §3 for mean margins if Gaussian assumptions are unsupported. Valid random-label p-values require sanctioned assignments under the relevant sharp null. Preserve rotations, repeats, captures and source restrictions in the exchangeability orbit. Use a registered randomized negative control when the existing label design lacks one. A paired sign-flip test needs actual randomization or a justified joint sign-symmetry null. A weak mean inequality alone supplies no such exchangeability.
406:For CTTC with \(m\) loops or pairs per capture, use the sensitivity proxy \(s_F^2=\sigma_u^2[\rho+(1-\rho)/m]+\sigma_e^2/(mS)\), after conditioning on the frozen seed panel. The model uses \(m=8\) for Q2 and \(m=4\) for Q3 and retains eight df. An extra session-by-seed term would add a component to \(s_F^2\). Use actual real dev covariance for deployment planning. The main table assumes 0.10 for the fraction of common group variance shared within captures. The JSON covers 0, 0.10 and 0.25, a direct historical-panel SD sensitivity, and the legacy Work E high-variance sensitivity. The latter uses a single aggregate variance envelope for both seed counts because Work E never decomposed it.
413:| CTTC nine captures, eight Q2 loops, \(\rho=.10\) | 8 | 72.26% / 96.93% | 81.81% / 96.78% | 1.398 / .942 | .01249 / .00947 |
414:| CTTC nine captures, four Q3 loop-pair groups, \(\rho=.10\) | 8 | 43.91% / 80.59% | 55.14% / 82.60% | 1.934 / 1.268 | .01689 / .01236 |
418:For the random-label checks, archive the assignment law and compute the upper-tail randomization p-value by the exact legal assignment orbit or the plus-one Monte Carlo rule in §3. A diagnostic shuffle of deterministic real labels is insufficient without a valid exchangeability argument. The existing split proof and CPU replay do not establish real-capture exchangeability or Stage 4 gate completion.
```

### Event #51: `item.completed` — `command_execution` (`item_48`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command 'Get-Content -Encoding utf8 research/stage4_math_and_design/M6/verify_m6.py | Select-Object -Skip 23 -First 28; Get-Content -Encoding utf8 research/stage4_math_and_design/M6/verify_m6.py | Select-Object -Skip 375 -First 97'
```

**Aggregated Output:**

```text
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
    "steps_per_arm": 384,
    "pilot_steps_per_arm": 69,
    "batch": 32,
    "microbatch": 16,
    "eval_batch": 64,
    "max_prefix_tokens": 512,
    "max_candidate_tokens": 16,
    "max_train_seconds_per_step": 2.5,
    "max_eval_seconds_per_logical_row": 0.03,
    "max_peak_vram_mib": 65000,
    "natural_lock_coverage_min": 0.95,
    "cttc_cyclic_source_guard_symbols": 4096,
    "pluto_split_guard_frames": 100,
    "probability_parity_max_abs": 1e-5,
    "geometry_max_defect": 1e-6,
    "geometry_rank_relative_tolerance": 1e-6,
def budget():
    d = Decimal
    cpu_rate, ram_rate, gpu_rate = map(d, ("0.0000131", "0.00000222", "0.000694"))
    arm_count = len(R2["arm_ids"])
    pilot_steps = R2["pilot_steps_per_arm"] * arm_count
    remaining_steps = (R2["steps_per_arm"] - R2["pilot_steps_per_arm"]) * arm_count
    step_time = d(str(R2["max_train_seconds_per_step"]))
    row_time = d(str(R2["max_eval_seconds_per_logical_row"]))
    mothers = len(R2["synthetic_cells"]) * R2["synthetic_blocks_per_cell"] * R2["synthetic_mothers_per_block"]
    synthetic_rows = mothers * 5 * arm_count
    pluto, cttc = R2["pluto"], R2["cttc"]
    pluto_rows = (4 * pluto["packets"] + 2 * pluto["q3_pairs"]) * arm_count
    cttc_rows = (4 * cttc["packets"] + 2 * cttc["q3_pairs"]) * arm_count
    stress_rows = (7 * 140 + 1080) * arm_count
    intervention_rows = 2 * (4 * mothers + pluto["packets"] + cttc["packets"])
    pilot_dev_rows = sum(R2["pilot_dev_rows_per_arm"].values()) * arm_count
    post_dev_rows = sum(R2["calibration_dev_rows_per_arm"].values()) * arm_count
    gpu = "A100-80GB"
    # ROM ceilings per work unit. Separate start/stop reserves cover billed idle and metering.
    rows = (
        ("cpu_ingest_weight_staging", "CPU", 1800, 1, 8, None, None, "cpu"),
        ("cpu_causal_frontend_dual_cache_pool", "CPU", 7200, 4, 16, None, None, "cpu"),
        ("cpu_audit_aggregate", "CPU", 1800, 2, 8, None, None, "cpu"),
        ("a100_initial_smoke_setup", gpu, 350, 2, 16, None, None, "pilot"),
        ("pilot_train_four_matched_arms", gpu, math.ceil(pilot_steps * step_time), 2, 16, pilot_steps, None, "pilot"),
        ("pilot_disjoint_dev", gpu, math.ceil(pilot_dev_rows * row_time), 2, 16, None, pilot_dev_rows, "pilot"),
        ("resume_same_four_arms", gpu, math.ceil(remaining_steps * step_time), 2, 16, remaining_steps, None, "arm_comparison"),
        ("post_fit_disjoint_calibration_dev", gpu, math.ceil(post_dev_rows * row_time), 2, 16, None, post_dev_rows, "arm_comparison"),
        ("blind_synthetic_core", gpu, math.ceil(synthetic_rows * row_time), 2, 16, None, synthetic_rows, "arm_comparison"),
        ("blind_pluto_raw", gpu, math.ceil(pluto_rows * row_time), 2, 16, None, pluto_rows, "pluto"),
        ("blind_cttc_raw", gpu, math.ceil(cttc_rows * row_time), 2, 16, None, cttc_rows, "cttc"),
        ("blind_real_stress", gpu, math.ceil(stress_rows * row_time), 2, 16, None, stress_rows, "controls"),
        ("blind_matched_slot_text_controls", gpu, math.ceil(intervention_rows * row_time), 2, 16, None, intervention_rows, "controls"),
        ("a100_remaining_setup_teardown", gpu, 750, 2, 16, None, None, "arm_comparison"),
        ("explicit_retry_reserve", gpu, 900, 2, 16, None, None, "retry"),
    )
    breakdown = []
    totals = {}
    for task, device, seconds, cores, gib, steps, eval_rows, group in rows:
        rate = (gpu_rate if device == gpu else d(0)) + cpu_rate * cores + ram_rate * gib
        cost = rate * seconds
        totals[group] = totals.get(group, d(0)) + cost
        breakdown.append({"task": task, "group": group, "device": device, "max_billed_seconds": seconds,
                          "max_physical_cores": cores, "max_gib": gib, "train_steps": steps,
                          "logical_eval_rows": eval_rows, "max_usd_per_second": str(rate), "max_usd": str(cost)})
    total, credit = d("9.50"), d("30.00") - d("17.12")
    planned = sum(totals.values()) - totals["retry"]
    reserved = sum(totals.values())
    unassigned = total - reserved
    a100_seconds = sum(row["max_billed_seconds"] for row in breakdown if row["device"] == gpu)
    expected_m4 = (d("0.00074684") * 10274 + totals["cpu"])
    archive = json.loads((DATA / "bench_steady.json").read_text(encoding="utf-8"))
    train = archive["train_bench"]["Q3_after_pilot_kv_mb16"]
    evaluation = archive["dev_eval_bench"]["Q3_after_pilot_eval_b64"]
    assert train["peak_vram_mb"] == 52972.5 and evaluation["peak_vram_mb"] == 45409.9
    assert expected_m4 == d("8.44073416")
    assert a100_seconds == 10987 and reserved == d("9.07079564") and unassigned == d("0.42920436")
    assert (planned, totals["cpu"], totals["pilot"]) == (d("8.39064764"), d("0.7677"), d("0.87663520"))
    assert (pilot_steps, remaining_steps) == (276, 1260)
    assert (synthetic_rows, pluto_rows, cttc_rows, stress_rows, intervention_rows) == (40960, 53760, 28800, 8240, 23744)
    assert sum(row["logical_eval_rows"] or 0 for row in breakdown) == 171504
    assert all(row["device"] in ("CPU", gpu) for row in breakdown)
    assert reserved + unassigned == total < credit and credit - total == d("3.38")
    return {
        "pricing_url": "https://modal.com/pricing", "only_gpu": gpu, "rows": breakdown,
        "stage3d_reference_bench": {"train_peak_allocated_mib_mb16": train["peak_vram_mb"],
                                    "eval_peak_allocated_mib_b64": evaluation["peak_vram_mb"],
                                    "train_seconds_per_step": train["steady_32ex_step_ms"] / 1000,
                                    "eval_seconds_per_row": evaluation["ms_per_ex"] / 1000},
        "group_usd": {key: str(value) for key, value in totals.items()},
        "m4_same_scope_at_12_gib_usd": str(expected_m4),
        "a100_16_gib_rate_delta_vs_m4_per_second": str(4 * ram_rate),
        "a100_seconds_including_retry": a100_seconds, "scheduled_usd_without_retry": str(planned),
        "scheduled_usd_with_retry": str(reserved), "unassigned_inside_cap_usd": str(unassigned),
        "total_reserve_inside_cap_usd": str(totals["retry"] + unassigned),
        "cpu_plus_pilot_usd_before_go": str(totals["cpu"] + totals["pilot"]),
        "cap_usd": str(total), "historical_credit_remainder_usd": str(credit),
        "buffer_usd_at_cap": str(credit - total), "required_workspace_net_spend_guard_usd": "0.00",
        "live_workspace_or_gpu_execution_verified": False, "spend_this_round_usd": "0.00",
    }


def disjoint_pilot_design(ledger):
    # Independent dev proxy and blind statistic conditional on effect, with zero early efficacy alpha.
    alpha = R2["domain_family_alpha"] / 3
    critical = norm.isf(alpha)
    slope, prior_var = 0.60, 4.0
    posterior_var = 1 / (1 / prior_var + slope**2)
    posterior_sd_blind = math.sqrt(1 + posterior_var)
    minimum_pp = R2["predictive_go_probability_min"]
    cutoff = (critical - norm.isf(minimum_pp) * posterior_sd_blind) / (posterior_var * slope)
    rng = np.random.default_rng(20261127)
    fixed_cost = float(ledger["scheduled_usd_without_retry"])
    cpu_cost = float(ledger["group_usd"]["cpu"])
    gpu_pilot_cost = float(ledger["group_usd"]["pilot"])
    full_after_pilot = fixed_cost - cpu_cost - gpu_pilot_cost
    strata = {}
```

### Event #52: `item.completed` — `command_execution` (`item_49`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command 'Get-Content -Encoding utf8 research/stage4_math_and_design/M3/ref_pool.py | Select-Object -Skip 20 -First 150; Get-Content -Encoding utf8 research/stage4_math_and_design/M3/ref_pool.py | Select-Object -Skip 205 -First 110; Get-Content -Encoding utf8 research/stage4_math_and_design/M3/ref_pool.py | Select-Object -Skip 355 -First 28'
```

**Aggregated Output:**

```text
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
        t = np.array([index(tick) - self.anchor_tick for tick in ticks], dtype=np.float64) / self.ticks_per_unit
        if np.any(t < 0) or not np.isfinite(t).all():
            raise ValueError("Clock predates the Brownian prediction anchor or has invalid units")
        cov = (self.diffusion_var_per_unit * np.minimum.outer(t, t)
               + self.v_omega * np.outer(t, t)
               + self.p_phi0 + self.cov_phi0_omega*(t[:, None]+t))
        return self.phi0_hat+self.omega_hat*t, cov


@dataclass(frozen=True)
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

    def logpdf(self, values):
        """Return (..., H, 4, N) log densities for observations (..., d)."""
        x = np.asarray(values, dtype=np.complex128)
        if x.ndim == 0 or x.shape[-1] != self.mean.shape[-1]:
            raise ValueError("Observation has the wrong dimension")
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite observation")
        d = x.shape[-1]
        out = []
        for mean, cov in zip(self.mean, self.covariance):
            l = np.linalg.cholesky(cov)
            residual = x[..., None, None, :] - mean
            whitened = solve_triangular(l, residual.reshape(-1, d).T, lower=True,
                                        check_finite=False).T.reshape(residual.shape)
            log_norm = d * math.log(math.pi) + 2*np.log(l.diagonal().real).sum()
            out.append(-np.sum(np.abs(whitened)**2, axis=-1)-log_norm)
        return np.stack(out, axis=-3)

    def project(self, weights):
        # The scored scalar is weights @ y, with no implicit conjugation.
        w = np.asarray(weights, dtype=np.complex128)
        cov = np.einsum("i,hij,j->h", w, self.covariance, w.conj()).real
        return GaussianLaw((self.mean @ w)[..., None], cov[:, None, None])


def conditional_gaussian(covariance, means, new, past, seen):
    """Condition on measured past coordinates, at each latent node."""
    k = covariance
    knn = k[:, new][:, :, new]
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
                raise ValueError(f"Observation unavailable at deadline: {raw.acquisition_id}")
            if raw.matched is None:
                if not allow_missing or raw.digest is None:
                    raise ValueError(f"Missing observation payload: {raw.acquisition_id}")
                sample = None
                digest = raw.digest
            else:
                sample = canonical_iq(raw.matched, self.frame)
                if raw.shares_channel and not np.isfinite(sample):
                    raise ValueError(f"Nonfinite valid acquisition: {raw.acquisition_id}")
                digest = sample_digest(sample)
                if raw.digest is not None and raw.digest != digest:
                    raise ValueError(f"Conflicting payload digest for {raw.acquisition_id}")
            obs = Observation(raw.clock, sample, raw.shares_channel, digest)
            previous = by_id.setdefault(obs.acquisition_id, obs)
            if previous != obs:
                raise ValueError(f"Conflicting copies of {obs.acquisition_id}")
        return tuple(by_id[key] for key in sorted(by_id))

    def _conditional(self, ids, boundary):
        past = [(self.indices[o.acquisition_id], o.matched) for o in boundary
                if o.acquisition_id in self.indices]
        return conditional_gaussian(self.covariance, self.means,
                                    [self.indices[k] for k in ids],
                                    [index for index, _ in past], [val for _, val in past])

    def update(self, state, records, as_of):
        self._validate_state(state)
        consumed = {r.acquisition_id: r for r in (*state.history, *state.added)}
        additions = {}
        for obs in self._canonical_records(records, as_of, allow_outside=True):
            prior = consumed.get(obs.acquisition_id)
            if prior is not None:
                if (prior.clock != obs.clock or prior.shares_channel != obs.shares_channel
                        or prior.digest != obs.digest):
                    raise ValueError(f"Conflicts with consumed acquisition {obs.acquisition_id}")
                # A retired payload has no likelihood, even if someone returns its sample.
            elif obs.shares_channel:
                if obs.acquisition_id not in self.indices:
                    raise ValueError(f"No validated provenance for {obs.acquisition_id}")
                if obs.clock.role == "history":
                    raise ValueError(f"History ID missing from the F0 ledger: {obs.acquisition_id}")
                additions[obs.acquisition_id] = obs
        if not additions:
            return state
        merged = {r.acquisition_id: r for r in state.added}
        merged.update(additions)
        if len(merged) > MAX_REFERENCES:
            raise ValueError("More than 64 distinct revealed valid acquisitions")
        ids = tuple(sorted(merged))
        accepted = tuple(merged[key] for key in ids)
        law = self._conditional(ids, state.history)
        ell = law.logpdf([obs.matched for obs in accepted])
        joint = state.base + ell
        logz = float(logsumexp(joint))
        return PoolState(self._owner, state.base, state.history, accepted,


def fixed_slots(canonical_codes, q2, q3):
    """Versioned canonical header + two always-present float32 evidence vectors."""
    if len(canonical_codes) != 3 or any(v not in range(4) for v in canonical_codes):
        raise ValueError("Expected S17 inner1, inner2 and relative turn")
    slots = readonly((q2, q3), "<f4")
    if (slots.shape != (2, 4) or not np.isfinite(slots).all() or np.any(slots < 0)
            or not np.allclose(slots.sum(axis=1), 1, atol=2e-7, rtol=0)):
        raise ValueError("Expected two normalized four-class evidence vectors")
    return readonly((WIRE_VERSION, *canonical_codes), np.uint8), slots
```

### Event #53: `item.completed` — `command_execution` (`item_50`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command 'Get-Content -Encoding utf8 research/stage4_math_and_design/M2/THEOREM_M2.md | Select-Object -Skip 339 -First 28'
```

**Aggregated Output:**

```text
|---|---|
| Arm matching | Match allowed observations and deadlines, frozen host, signed text, candidate strings, quotient/lift when isolating fusion, calibration opportunity, score floor, training manifest, optimizer and step budget. Zero uses the prescribed coupled-phase input ablation. Naive-Additive shares the inputs with a centered additive code/phase score in canonical coordinates. Verify its mixed code-by-phase term is zero. Changes to the quotient, head or training recipe create a package comparison. A fusion attribution requires the matched ablation. |
| Common absolute gate | Require NLL improvement \(>0\) over uniform and the dev prior. Require accuracy \(>1/K+0.02\) and improvement \(>0\) over the frozen prior classifier. Test both skill statistics against a justified locked random-label null. Require paired Brier improvement \(>0.005\) over the dev-selected lowest-Brier baseline from uniform, dev prior and matched Zero, and \(>0\) over every required baseline in that set apart from the candidate. Require a valid blind ECE upper bound \(<0.05\) on ten frozen top-label bins. Require native equivariance and projection idempotence max defect \(\le10^{-5}\) on the declared group support. |
| NIG | Lock a same-information oracle or independently fitted physical reference. Establish a positive denominator analytically or with a valid paired lower bound as an additional IUT endpoint, and require \({\rm LB}(0.90H(Y)-L(q_X)+0.10L(q_{\rm ref}))>0\). A balanced synthetic target law has \(H(Y)=\log K\). For real data predeclare the target class law or carry entropy uncertainty jointly. A causal approximate reader yields reference-normalized wording. An indefensible denominator makes the gate ineligible. |
| Arm 1 host-pure extrapolation | Against each of Zero and Naive-Additive require \({\rm LB}(D_L)>0.02\) nats and \({\rm LB}(D_A)>0.02\) on the locked extrapolation mixture, plus common absolute/NIG gates. Score the frozen host's candidate outputs with quotient alignment and scalar calibration. Coupled physical features can enter the soft tokens. Verify the output graph and absence of a direct physical logit path. |
| Arm 1 fixed-text intervention | At the same checkpoint, waveform/cache, signed text, scorer and scalar, require \({\rm LB}(D_L(\mathrm{matched},\mathrm{slot\ swap}))>0\) and \({\rm LB}(D_L(\mathrm{matched},\mathrm{slot\ shuffle}))>0\). The signed swap flips the slots. The shuffle is a separately seeded, frozen mapping of slot values across signs within prespecified absolute-CFO, lag, SNR and availability strata, without labels. Record its effective mismatch fraction and the locked treatment of strata that cannot be shuffled. Keep signs/rotations with their mothers. Audit canonical centered code-by-phase mixed contrasts with the additive arm as the zero-interaction control. This establishes a response to the supplied slot. Waveform-based CFO estimation is a different hypothesis. |
| Arm 2 host residual value | With \(q_2=\operatorname{softmax}((\ell_{\rm phys}+\gamma\tanh\ell_{\rm host})/T_2)\), require \({\rm LB}(D_L)>0.02\) nats and \({\rm LB}(D_A)>0\) against independently dev-calibrated Direct Physical with identical evidence and representation. Also require \({\rm LB}(D_L)>0\) versus the same-stack residual-off forecast \(\operatorname{softmax}(\ell_{\rm phys}/T_2)\). Apply both slot interventions only in the host residual branch. Require positive NLL lower bounds for both while \(\ell_{\rm phys}\) is bit-identical and \(\gamma,T_2\) remain fixed. Apply common absolute/NIG gates to confirmatory Arm 2 release. This tests incremental host value over the specified physical reader. Arm 1 evidence must come from Arm 1. |
| Calibration | Freeze one dev-fitted scalar per checkpoint/arm and task or predeclared stratum, using the same predeclared search range and objective. Hash the calibrator, priors and adapter with the scores. Reuse the treatment's scalar for its ablations. A synthetic-to-real zero-shot route transfers the synthetic-dev fit and permits only registered causal sample state. A real-dev fit is a separate real-calibrated route and excludes the held-out capture. Choose the confirmatory route on dev. Oracle blind temperature and calibration sensitivity are retrospective diagnostics. |
| Q2 gain | Under the same frozen RefSufficientPool scorer require \({\rm LB}(L_{\rm one}-L_{\rm two})>0.02\) nats and \({\rm LB}(L_{\rm none}-L_{\rm one})>0\). One uses the registered first past release. Two includes the distinct second valid release. Q2 two-valid clears its absolute and reference-normalized gates. Report Direct Q2, accuracy/SER, bit NLL and EVM as secondary comparisons. |
| Q2 exact controls | Valid must give byte-identical preprocessing, fixed slots, masks, positions, candidate context and deterministic scorer outputs after appending a raw duplicate ID, independent-channel record or empty padding, or permuting the reference list. Require 0.0 max probability change, TV and KL for each pair. Verify canonical reuse at the real API boundary where used. Validate ID and samples before deduplication and fail conflicting IDs. A genuinely fresh acquisition with a new ID enters the two-valid efficacy test; superiority is expected across cases. |

With metadata \(M\) and authorized references \(R\) held fixed, a deterministic waveform cache \(Z\) and host forecast \(Q\) satisfy \(I(Y;Q\mid R,M)\le I(Y;Z\mid R,M)\le I(Y;W\mid R,M)\). The Stage 3D loss ladder compares finite readouts and includes their regret. Conditional entropy gaps require Bayes readouts at each named information set. The residual-off and fixed-physical interventions compare architectures at the same information set. [M1 arm design](../discussions/M1/prompt_r2.md) · [M3 reference contract](../discussions/M3/prompt_r2.md) · [M6 DPI](../M6/THEOREM_M6.md)



### Three-claim testing tree

Freeze one primary route and checkpoint panel per claim before blind access. Claim 1 defaults to Arm 1. A preregistered Arm 2 substitution requires the complete residual-value, absolute, Q2 and structural endpoints and yields an Arm 2 claim. Both arms can instead be required jointly by taking the maximum over both full endpoint lists. Independent confirmatory releases for either arm require extra hypothesis slots and revised multiplicity before acquisition.

| Claim | Mandatory endpoints and scope | Unit |
|---|---|---|
| 1. Synthetic extrapolation + Q2 | Require the locked extrapolation route's arm-specific, absolute, NIG, calibration and structural endpoints. Also require the Q2 ordered gain, absolute and exact controls. Use the M4 32-block equal-weight mixture of eight signed large-angle settings with four blocks per setting, or preregister a 131-block design with the same target mixture weights and the exact per-stratum weights/counts. Region-wide efficacy means the prespecified average. Per-setting claims are descriptive unless included in the IUT. The 131-block homogeneous G1 curve below is a power proxy, rather than a replay of that stratified design. | Independent generator blocks. Keep a mother's repeats and controls paired. Use the fixed-stratum weighted contrast where applicable. |
| 2. Pluto | Require the common absolute/calibration/structure and ordered Q2 gate on the natural rows. Release pilots 15 and 18. Include every registered lock failure with the common uniform fallback. Report lock coverage, stress results on 140 base frames, and Q1/Q3 diagnostics. | 2,240 frames in 16 chronological blocks from one campaign, with copies nested. Condition on campaign. Exact inference needs a valid conditional block model or a valid design randomization. If shared temporal dependence invalidates the model, report the campaign descriptively. |
| 3. CTTC + f708_snr10 | Require the same paired natural-data Q2, absolute and exact controls, pilots 126 and 127, and the single-capture transfer guard on f708_snr10. Require \(L_{\rm two}<L_{\rm one}<L_{\rm none}\) and \(L_{\rm two}<L_{\rm prior}\) as fixed-cohort point inequalities there, complete reporting and zero integrity failures. Exclude this capture from checkpoint fitting, calibration and nuisance/reader/prior/hyperparameter tuning. Its causal sync and authorized pilots remain permitted. | 1,440 packets in 72 loops nested in nine captures, and 720 Q3 pairs in 36 disjoint loop-pair groups nested in the same captures. Average paired effects within capture, then weight captures equally. Report the held-out capture separately, plus leave-out results for the other eight. The holdout's p=0/1 bookkeeping is a cohort guard, with no standalone population test. |

Real primary Q2 uses paired symbol NLL for its ordered-evidence contrasts. Q1 relative-turn, real Q3 retransmission and stress metrics are secondary unless a full confirmatory endpoint set is appended to the relevant slot before scoring. Preserve every failed-lock row and apply the same fallback rules across arms. Validate identities across the regular and fallback paths. At each release, audit raw support and timestamp using the locked causal alignment and without consulting target labels or unreleased references.

CTTC replays one public cyclic TX sequence. Disjoint roles/splits prevent direct observation reuse. The common fixed test range \([334096,455000)\) can still induce source-ordinal dependence across captures. Scope the primary cohort estimand to the registered captures and fixed source. Justify independent capture innovations given the locked conditioning before nine-unit inference. If common residual randomness remains across captures and there is no valid design-based alternative, report descriptive results. Show source-ordinal aggregation, capture leave-one-out and capture-by-source sensitivity. The 36/72 nested groups contribute to capture means. They do not supply independent capture or new-source draws. [M4 benchmark and split](../M4/THEOREM_M4.md#2-hardware-benchmark-and-causality-contract)
```

### Event #54: `item.completed` — `command_execution` (`item_51`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "Get-Content -Encoding utf8 research/stage4_math_and_design/M6/verify_m6.py | Select-Object -Skip 45 -First 42; Get-Content -Encoding utf8 research/stage4_math_and_design/M6/THEOREM_M6.md | Select-Object -Skip 159 -First 14; rg -n 'ledger|futility' research/stage4_math_and_design/M6/results_m6.json | Select-Object -First 8; rg -n 'stage3d.*MB|peak_vram_mb' research/stage3d_qwen_signed_phase/modal_runner.py | Select-Object -First 12"
```

**Aggregated Output:**

```text
    "natural_lock_coverage_min": 0.95,
    "cttc_cyclic_source_guard_symbols": 4096,
    "pluto_split_guard_frames": 100,
    "probability_parity_max_abs": 1e-5,
    "geometry_max_defect": 1e-6,
    "geometry_rank_relative_tolerance": 1e-6,
    "geometry_min_singular_ratio": 0.05,
    "geometry_max_relative_projection_residual": 0.10,
    "geometry_max_relative_ood_rmse": 0.25,
    "arm2_gamma": 0.15,
    "probability_floor": 1e-6,
    "domain_family_alpha": 0.05,
    "power_floor_each_joint_domain_claim": 0.90,
    "predictive_go_probability_min": 0.10,
    "absolute_chance_accuracy_increment": 0.02,
    "zero_accuracy_increment": 0.02,
    "zero_nll_increment": 0.02,
    "synthetic_1_minus_0b_nll_increment": 0.02,
    "real_two_minus_one_nll_improvement": 0.02,
    "cache_iq_nll_regret_ceiling": 0.05,
    "host_readout_nll_regret_ceiling": 0.05,
    "arm2_host_nll_increment_vs_physical": 0.005,
    "fixed_text_slot_nll_increment": 0.005,
    "brier_increment": 0.005,
    "ece_upper": 0.05,
    "ece_bins": 10,
    "nig_eta": 0.10,
    "pluto": {"packets": 2240, "q3_pairs": 2240, "clusters": 16,
              "reference_indices": [15, 18], "q3_lag_symbols": 126},
    "cttc": {"packets": 1440, "q3_pairs": 720, "clusters": 72, "q3_loop_pairs": 36,
             "sessions": 9, "holdout": "f708_snr10", "holdout_packets": 160, "holdout_q3_pairs": 80,
             "reference_indices": [126, 127], "q3_lag_symbols": 500000},
    "synthetic_blocks_per_cell": 8,
    "synthetic_mothers_per_block": 64,
    "synthetic_cells": ["G1", "G2", "G3", "G4"],
    "synthetic_primary_cells": ["G1", "G3"],
    "synthetic_primary_endpoint": "Q3 repeat pre-reveal",
    "synthetic_geometry_abs_omega_h_ell": {"G1": [0.40, 2, 2], "G2": [0.20, 2, 2],
                                          "G3": [0.20, 4, 4], "G4": [0.40, 1, 1]},
    "synthetic_ref_times_in_h_units": [0.25, 0.75],
    "pilot_dev_rows_per_arm": {"synthetic": 500, "pluto": 250, "cttc": 250},
    "calibration_dev_rows_per_arm": {"synthetic": 1500, "pluto": 750, "cttc": 750},

For evidence $D$ and remaining budget $b$, let $U(D)$ be expected utility of the best terminal decision. The finite allocation law is $V(D,b)=\max\{U(D),\max_{j:c_j\le b}[-c_j+E V(D,X_j,b-c_j)]\}$ over eligible jobs $j$. This justifies taking free information before an otherwise identical paid decision and prices an early stop through future avoidable cost. With linear compute cost, the model below gives $E[C]=C_{\rm CPU}+P(\mathrm{screen})C_{\rm pilot}+P(\mathrm{GO})(C_{\rm full}-C_{\rm CPU}-C_{\rm pilot})$. Separate branch budgets and the worst-case cap constrain which alternatives enter the recursion.

Buy exactly **69 of 384 steps per arm (17.96875%)**, then 1,000 pilot-dev rows per arm. Training, pilot dev, post-fit calibration dev, and blind data are disjoint by mother and eligible physical/source support. Exclude `f708_snr10` from fit, shared-prior tuning, pilot and calibration. Preserve checkpoint, optimizer, RNG and data position for resume. The pilot spends **zero efficacy alpha**. Continue only when the registered independent dev model gives predictive probability $\ge.10$ of clearing the synthetic joint gate, the engineering predicates pass, and powered enrollment is justified. Each real branch also requires its own predictive probability $\ge.10$ before its blind rows are funded. Calibrate the map from 69 steps to final effects on disjoint development evidence. A step fraction supplies no statistical information fraction. When the law or enrollment is unresolved take HOLD for the confirmatory claim. An unfunded branch contributes $p=1$ to the family.

Use **8 independent 64-mother blocks per G1–G4 cell**, four per sign on new mother namespaces. Use the Stage 3D CFO/lag geometry $(|\omega|,h,\ell)$ of $(.40,2,2),(.20,2,2),(.20,4,4),(.40,1,1)$ respectively, at 2 dB and diffusion SD .10 per square-root symbol. Synthetic primary is Arm 1 versus Arm 0b on **Q3-repeat pre-reveal** for the equal-weight large-angle G1/G3 mixture. G2/G4, per-cell intervals, Q3 post-reveal and Q2 are diagnostics. Five forecasts per mother are Q3 pre/post and Q2 none/one/two, with independent acquisition noise for references at $h/4,3h/4$. Collect them before W2 and withhold their payloads from the initial forecast. Commit the two Q3 forecasts before the repeat arrives at $h+\ell$. The four synthetic intervention views use the Q3 pre-reveal score. These reference times are the new diagnostic design.

For each arm allocate pilot-dev rows as 500 synthetic (100 mothers × 5), 250 Pluto and 250 CTTC (125 Q2 one/two pairs each). Allocate post-fit dev rows as 1,500 synthetic (300 × 5), 750 Pluto (125 × 6) and 750 CTTC (150 × 4 plus 75 repeats × 2). Balance synthetic dev across the four cells and counterbalance signs, using disjoint mother groups. Freeze cohort IDs and prove adequate support or stop the corresponding claim. Real primary is Q2 two-minus-one. Choose Arm 1 if qualified by the frozen dev rule, otherwise Arm 2 if qualified including physical-increment safeguards. Commit before blind access. The alternative arm's estimates are descriptive. Fit scalar temperatures on post-fit dev, then apply the score floor $q^\epsilon=(1-4\epsilon)q+\epsilon$, $\epsilon=10^{-6}$, once across forecasts and comparators.

Use one-sided bounds with a valid predeclared cluster model. For each domain, take the maximum of component p-values (intersection-union test). Apply Holm across synthetic, Pluto and CTTC at .05, with sorted thresholds $.05/3,.05/2,.05$. All relevant inequalities below are strict for positive claims.

- Require NLL lower improvement over uniform and frozen dev prior $>0$ and over Zero $>.02$. Require accuracy lower improvement over chance and Zero $>.02$, over frozen majority/prior $>0$. For synthetic require Arm 1 over 0b NLL lower improvement $>.02$. For real require two versus one Q2 NLL lower improvement $>.02$. All registered failures get a common uniform score.
- For baseline loss $L_b$ and matched physical comparator $L_o$, require positive lower bounds on $L_b-L_o$ and $.90L_b-L_q+.10L_o$. This is reference-normalized when an oracle is uncertified. Use $L_b=\log 4$ under uniform synthetic labels and a frozen causal prior on real. Require upper candidate NLL excess $<.05$ versus the matched physical cache and the finer IQ comparator wherever permissions match the stated release target. Do not require a learner to beat the Bayes risk.
- Require Brier improvement lower bound $>.005$ against uniform, dev prior and Zero, and ten-bin ECE upper bound $<.05$ through a joint signed-bin region or 1,024 linear sign constraints. Require preregistered random-label accuracy/NLL tests where design exchangeability is justified. For real prep a valid randomized negative control; an unvalidated random-label diagnostic cannot support confirmation.
682:    "predictive_futility_threshold": 0.1,
686:    "futility_z": -0.2704380422162607,
714:        "futility_count": 348,
735:        "futility_count": 545,
756:        "futility_count": 93,
465:                    "peak_vram_mb": round(pk, 1),
494:                "peak_vram_mb": round(pk, 1),
```

### Event #55: `item.completed` — `command_execution` (`item_52`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "Get-Content -Encoding utf8 research/stage4_math_and_design/M3/ref_pool.py -TotalCount 22; Get-Content -Encoding utf8 research/stage4_math_and_design/M6/THEOREM_M6.md | Select-Object -Skip 175 -First 29; rg -n 'blind_matched|Round 2|modal_budget|a100_plan|execution_budget' research/stage4_math_and_design/M6/results_m6.json | Select-Object -First 8; rg --files research/stage4_math_and_design/M1 -g 'reply_r2.md' -g '*.py'"
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
Pluto's 16 blocks are chronological in one campaign. Claim block inference only with a justified dependence model; otherwise describe that cohort. CTTC has 72 loops nested in nine sessions and 36 disjoint loop pairs for Q3. Keep seed, rotations and repeated rows within the physical units; account for repeated source ordinals as well as capture dependence. Report the separate 160-packet/80-Q3-pair result on held-out `f708_snr10`. Nine sessions constitute a benchmark cohort, with one wholly unseen capture. Conditional fixed-seed claims require no population-of-seeds variance estimate. Use worst first-Holm alpha and either registered joint power $\ge.90$ or endpoint failure allocations summing to $\le.10$. The 32-block design does not inherit the 131-block result. When a confirmatory power target fails, retain a descriptive-only route if predeclared, or stop before GPU. Additional seeds or paid blocks require new authorizations and design; exact output reuse on new, independent compatible mothers must be proved at the complete prefix key.

With preblind information $\mathcal D$, GO indicator $G$, and valid conditional final p-value $p$, $P(G=1,p\le a)=E[1_G P(p\le a\mid\mathcal D)]\le a$. Thus a dev-only futility choice preserves Type-I control, including dev-only choice of real candidate. Holm accepts dependence across the three valid domain p-values. Early efficacy from an unvalidated pilot/final correlation would invalidate this argument. The new replay models independent $X\sim N(.6\mu,1)$ (dev) and $Y\sim N(\mu,1)$ (blind) conditional on $\mu$, using a $N(0,4)$ planning prior. The posterior variance is $v=(.25+.6^2)^{-1}$ and mean $.6vX$. Predictive $Y$ has variance $1+v$. At the first Holm boundary and predictive threshold .10 the illustrative dev cutoff is $X\ge .0467962$. It is a model diagnostic; actual promotion uses the independently validated joint model.

The 10,000-candidate replay shows mean scheduled cost **\$3.357607 vs \$8.390648** (**59.9839%** simulated saving, **59.8771%** integrated), **94.040%** working endpoint power (**94.1785%** integrated), and **0.7333%** boundary-null promotion (**0.7220%** integrated), under the modeled mixture, effect sizes and dev slope. CPU costs are charged even on screen rejection. The fixed single-endpoint boundary alpha is 1.6667%. Round 1's separate parametric O'Brien–Fleming sensitivity replay still returns 64.3619% cost saving and 4.4695% binding Type-I error. Production joint power and the pilot mapping require independent development receipts. [Results](results_m6.json)

### 4.4 The sole Stage 4 Modal ledger

Use **A100-80GB for all GPU work**, as preemptible Functions, at **\$0.00075572/s** including 2 physical cores and 16 GiB. Use hard CPU/memory request-limit pairs and no disk requests that raise the charged memory amount. Modal bills the greater of requested and used resources. M4's **\$8.44073416** includes 10,274 A100 seconds at **\$0.00074684/s** with 12 GiB, plus \$0.76770000 CPU-only preparation. Its 131,760 blind rows exclude 16,000 dev rows. Moving to 16 GiB adds **\$0.09123312** on the same seconds. Round 2 adds 713 A100 seconds for fixed-position counterfactuals. Stage 3D measured **52,972.5 MiB** peak allocated memory in training (microbatch 16) and **45,409.9 MiB** in eval (batch 64). Its source divides bytes by $1024^2$ even though the bench JSON says MB. Use the same hardware class; qualify the new longest cases before continuing. [M4 ledger](../M4/THEOREM_M4.md#33-budget-arithmetic-and-zero-bill-gate), [Stage 3D bench](../../stage3d_qwen_signed_phase/results/bench_steady.json), [Modal rates](https://modal.com/pricing), [Resource controls](https://modal.com/docs/guide/resources)

| Debit | Units | Seconds | USD ceiling |
|---|---:|---:|---:|
| CPU ingest and weight staging to Volume, 1 core / 8 GiB | Manifests and authorized data | 1,800 | 0.05554800 |
| CPU causal R3 frontend, dual cache, readers and pool, 4 cores / 16 GiB | One shard per source | 7,200 | 0.63302400 |
| CPU audit, preparation and aggregation, 2 cores / 8 GiB | Locked receipts | 1,800 | 0.07912800 |
| Initial A100 setup and longest-shape smoke | Before pilot | 350 | 0.26450200 |
| A100 pilot train | 4 × 69 steps | 690 | 0.52144680 |
| A100 pilot-dev score | 4,000 rows | 120 | 0.09068640 |
| A100 finish four arms at the same seed | 4 × 315 steps | 3,150 | 2.38051800 |
| A100 post-fit disjoint dev | 12,000 rows | 360 | 0.27205920 |
| A100 synthetic core | 40,960 blind rows | 1,229 | 0.92877988 |
| A100 Pluto natural | 2,240 frames / 53,760 blind rows | 1,613 | 1.21897636 |
| A100 CTTC natural | 1,440 packets + 720 Q3 pairs / 28,800 blind rows | 864 | 0.65294208 |
| A100 real stress | 8,240 rows | 248 | 0.18741856 |
| A100 fixed-position slot/text controls | 23,744 rows | 713 | 0.53882836 |
| A100 remaining setup, commit and teardown | Billed overhead | 750 | 0.56679000 |
| **Scheduled without retries** | **10,087 A100 and 10,800 CPU-only seconds** | | **8.39064764** |
| Explicit controlled retry reserve | A100 | 900 | 0.68014800 |
| **Scheduled with retry** | **10,987 A100 and 10,800 CPU-only seconds** | | **9.07079564** |
848:  "modal_budget": {
997:        "task": "blind_matched_slot_text_controls",
research/stage4_math_and_design/M1\verify_m1.py
```

### Event #56: `item.completed` — `agent_message` (`item_53`)

M6’s Round 2 handoff now specifies a matched 69-step pilot for each arm at the same seed. I’ll align M4 to that shared ledger. It costs $0.87663520 before continuation and measures all four arms on the same dev screen. The $0.49121800 figure remains the earlier single-arm reference.

### Event #57: `item.completed` — `file_change` (`item_54`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M4\\verify_m4.py",
    "kind": "update"
  }
]
```

### Event #58: `item.completed` — `file_change` (`item_55`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M4\\verify_m4.py",
    "kind": "update"
  }
]
```

### Event #59: `item.completed` — `file_change` (`item_56`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M4\\verify_m4.py",
    "kind": "update"
  }
]
```

### Event #60: `item.completed` — `file_change` (`item_57`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M4\\THEOREM_M4.md",
    "kind": "update"
  }
]
```

### Event #61: `item.completed` — `command_execution` (`item_58`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n -i 'text.only|four synthetic|slot.flip|control per|shuffle' research/stage4_math_and_design/M6/THEOREM_M6.md | Select-Object -First 12"
```

**Aggregated Output:**

```text
131:At residual zero, $\operatorname{softmax}(\log p_0)=p_0$ before the shared scoring floor. Each pairwise residual log-odds shift has magnitude $\le .30$. A wrong physical leader with a gap greater than .30 cannot be reversed. Expanding the normalizer also bounds each single-case NLL gain by .30. Measure Arm 2 versus physical-only and versus shuffled residual with **the same physical prior vector on each row**. Measure Arm 1's native score independently. M1's CPU hybrid reached 64.62% and tied the physical comparator. M2's retrospective blind-temperature audit reversed Stage 3D's G1 phase NLL lead. Fit calibration on disjoint dev folds, then freeze it. [M1 results](../M1/THEOREM_M1.md), [M2 forensic audit](../M2/THEOREM_M2.md)
133:Score four alternate synthetic views for both 1 and 2: slot-only sign flip, text-only sign flip, both flips, and cyclic slot shuffle under a fixed seed within comparable strata with no fixed points. Reuse the all-matched scores. Fix waveform, labels, causal metadata, candidate IDs, slot count and positions. Lock equal-length text alternatives so the text-only intervention changes only that evidence path. Hold Arm 2's prior and label frame fixed. On real data add one fixed-text Q2-two reference-payload shuffle per record for both arms, preserving positions and observation masks. The counterfactuals test within-checkpoint behavior and contribute no observation likelihood factors.
165:Use **8 independent 64-mother blocks per G1–G4 cell**, four per sign on new mother namespaces. Use the Stage 3D CFO/lag geometry $(|\omega|,h,\ell)$ of $(.40,2,2),(.20,2,2),(.20,4,4),(.40,1,1)$ respectively, at 2 dB and diffusion SD .10 per square-root symbol. Synthetic primary is Arm 1 versus Arm 0b on **Q3-repeat pre-reveal** for the equal-weight large-angle G1/G3 mixture. G2/G4, per-cell intervals, Q3 post-reveal and Q2 are diagnostics. Five forecasts per mother are Q3 pre/post and Q2 none/one/two, with independent acquisition noise for references at $h/4,3h/4$. Collect them before W2 and withhold their payloads from the initial forecast. Commit the two Q3 forecasts before the repeat arrives at $h+\ell$. The four synthetic intervention views use the Q3 pre-reveal score. These reference times are the new diagnostic design.
174:- Slot use requires matched-vs-shuffled fixed-text NLL lower improvement $>.005$ and synthetic matched-vs-flipped $>.005$. Include these in the IUT when claiming the mechanism. A promoted Arm 2 must also clear $>.005$ over its residual-zero and fixed-prior shuffled-residual controls. Physical-only parity leaves the incremental host claim unresolved.
210:Count four answers per logical row, Q1 plus three Q2 forecasts per natural record and two Q3 forecasts per eligible repeat. Controls are four synthetic alternatives × 2,048 mothers × two arms and one real fixed-text Q2-two shuffle × 3,680 records × two arms. Physical-only and dual-cache reads run on CPU. There are 155,504 blind logical rows with interventions, and 171,504 including dev. The 2,477-s raw-real line from M4 divides into 1,613 s Pluto and 864 s CTTC. The new scope buys four matched checkpoints using 1,536 total steps. The Round 1 one-arm 2,880-step plan has been replaced.
```

### Event #62: `item.completed` — `file_change` (`item_59`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M4\\THEOREM_M4.md",
    "kind": "update"
  }
]
```

### Event #63: `item.completed` — `command_execution` (`item_60`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n 'Arm 0b|Naive.Additive|no.*interaction|mixed' research/stage4_math_and_design/M6/THEOREM_M6.md | Select-Object -First 12; rg --files research/stage4_math_and_design/M1 -g 'reply_r2.md'; rg -n 'Arm 0b|logit additive|no.*interaction|additive' research/stage4_math_and_design/M1/THEOREM_M1.md | Select-Object -Last 12"
```

**Aggregated Output:**

```text
90:**Additive obstruction.** For centered scores $f(z,\phi)=a(z)+b(\phi)$, the mixed rectangle $f(z,\phi)-f(z',\phi)-f(z,\phi')+f(z',\phi')$ is identically zero. For complex evidence (u(z)) and phase $\phi=\omega t$, score $\Re[u(z)e^{i\phi}\bar r_a]$ for class (a). The physical rectangle equals $\Re[(u(z)-u(z'))(e^{i\phi}-e^{i\phi'})\bar r_a]$, generically nonzero. At least one corner must have error at least one fourth of the rectangle norm by the triangle inequality. Build $u\cos\phi,u\sin\phi$ into the connector, or transport complex evidence first and apply a joint score. The rotation law then extrapolates for all $\omega t$ where the physical model holds. The verified toy probe fits on $|\omega|\le .25$ and evaluates at $\omega=\pm .40,t=4$ ($|\phi|=1.60>\pi/2$). Additive vs coupled OOD centered-logit RMSE is **.935161 vs .000000000000000738**, with rectangle L2 norm **3.998294** and subsecond execution. A smaller randomized frozen host can test software geometry; it has no guaranteed approximation to Qwen. A flexible nonlinear function plus a smooth bump supported outside the training interval can fit the same train data and give any OOD value. Some inductive bias or explicit mechanistic law is therefore indispensable for certified extrapolation.
125:| 0b · Naive-Additive | Same scaffold with separate populated phase slots. Frozen nonlinear attention may mix them. The additive obstruction in §3 applies to centered additive logits and their fixed tangent analogue. |
165:Use **8 independent 64-mother blocks per G1–G4 cell**, four per sign on new mother namespaces. Use the Stage 3D CFO/lag geometry $(|\omega|,h,\ell)$ of $(.40,2,2),(.20,2,2),(.20,4,4),(.40,1,1)$ respectively, at 2 dB and diffusion SD .10 per square-root symbol. Synthetic primary is Arm 1 versus Arm 0b on **Q3-repeat pre-reveal** for the equal-weight large-angle G1/G3 mixture. G2/G4, per-cell intervals, Q3 post-reveal and Q2 are diagnostics. Five forecasts per mother are Q3 pre/post and Q2 none/one/two, with independent acquisition noise for references at $h/4,3h/4$. Collect them before W2 and withhold their payloads from the initial forecast. Commit the two Q3 forecasts before the repeat arrives at $h+\ell$. The four synthetic intervention views use the Q3 pre-reveal score. These reference times are the new diagnostic design.
research/stage4_math_and_design/M1\reply_r2.md
45:**Proof.** Every code orbit intersects the canonical slice once. Applying equivariance to that representative proves necessity. The shift rule is additive, so \(R_jR_k=R_{j+k}\) proves sufficiency. A deterministic frozen host can be part of \(F\), because its inputs are identical across the orbit. Normalize in the canonical order and gather by integer indices to obtain bitwise equality for identical canonical computations. With an explicit common pilot frame, canonicalize that evidence and use the extended Q2 shift. For Q3 independent, return exact quarters. \(\square\) [Stage 3C canonicalization and readout](../../stage3c/G/quotient.py)
146:With \(p_i=\lambda_i/\sum_j\lambda_j\), the effective rank is \(\exp(-\sum_i p_i\log p_i)\) and the stable rank is \(\sum_i\lambda_i/\lambda_1\). An arbitrary rank-one matrix would retain at most \(50.19\%\) of the physical Frobenius energy and leave a relative Frobenius residual of \(0.70576\). The host phase contrast has \(3.52\%\) of the physical Frobenius norm and relative Frobenius error \(0.99076\). Work F's orientation gain is its normalized Frobenius projection onto \(M\). Its factorial effect instead uses a paired-mother measure and a signed contrast scaled by \(1/2\). A probability-spectrum rank is different from attention rank or additive-logit rank, since softmax is nonlinear.
269:The entries are additive output shifts. A realized independent sector label in an RX rotation can roll even though the predictive vector remains uniform. CFO and timestamps are unchanged by a constant gauge. A phase jump, changing an LO slope, or physically applying a new transmitter waveform defines a different intervention.
```

### Event #64: `item.completed` — `command_execution` (`item_61`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "Get-Content -Encoding utf8 research/stage4_math_and_design/M1/reply_r2.md -TotalCount 50; rg -n 'Arm 0b|PhaseContext|Naive' research/stage4_math_and_design/M1/THEOREM_M1.md | Select-Object -Last 12"
```

**Aggregated Output:**

```text
# M1 Round 2 handoff

The updated verifier exited **0** under the prescribed Python 3.12 `uv run` command. All learned configurations used the same frozen isotropic host, 9,064 trainable parameters, and three matched seeds. Temperatures were fit only on the disjoint dev split. [Methods and contracts](THEOREM_M1.md#round-2-convergence) · [Verifier](verify_m1.py) · [Numerical receipt](results_m1.json)

| Cell | Arm 0 · Separate | Arm 1 · Host-Pure | Arm 2 · Physical+Residual | Arm 3 · Physical only |
|---|---:|---:|---:|---:|
| G1 | 62.16% / 1.04094 | 64.60% / 0.97853 | 64.62% / 0.94222 | 64.62% / 0.94113 |
| G2 | 45.21% / 1.05384 | 46.81% / 1.02343 | 46.78% / 1.02174 | 46.78% / 1.02223 |
| G3 | 58.80% / 1.07606 | 61.91% / 1.01301 | 62.11% / 0.98098 | 62.11% / 0.97957 |
| G4 | 45.18% / 1.05790 | 46.97% / 1.02097 | 47.75% / 1.01639 | 47.75% / 1.01585 |

Table entries are accuracy and NLL in nats. Arm 3 uses unit temperature.

- **Host-Pure clears 55% on G1 and G3 in every seed.** Arms 1 and 2 have zero grouped equivariance defect and worst ungrouped defects of \(1.49\times10^{-7}\) and \(5.96\times10^{-8}\). Arm 1's G1 fixed-text sign swap and balanced shuffle add 2.0172 and 1.0072 nats. Its zero-slot retrain drops to 15.38% on G1 and 16.80% on G3.
- **Keep quotient sharing in the Stage 4 design.** It improves NLL at both coupling settings in all four CPU cells. The explicit products increase G1 NLL by 0.00860 nats with the quotient held fixed, with a descriptive paired 95% interval of \([0.00733,0.00988]\). Carry the quotient-separate comparator into the Qwen test.
- **Keep physical-only as an independent baseline.** The residual leaves top-1 unchanged in all cells and raises G1 NLL by 0.00109 nats against Arm 3. Its fixed-text slot effects are unresolved. Require a paired incremental gain before making the residual a decision path.
- **M6** — Native centered candidate Jacobians all have rank 3, and the shared final-projector dev stacks have rank 192. On G1, Arm 1's median native oracle-pair margins are 0.522–0.785 across seeds. Run the same native candidate and fixed-text diagnostics at the actual Qwen answer interface before applying M2's promotion gate.
- **M3, M4, and M5** — The 16-bit payload keeps `s17 = word >> 12` bit-exact. A common RX turn fixes Q1 and Q2 and rolls received sectors. A separate TX relabeling shifts the Q2 chart. Marginalize M3's normalized joint over sector or TX label as required, with the receive-frame lift applied only to Q3. Integrate one shared CFO latent across clocks. Synthetic R3 checks pass for both packet formats, including all four turns. The run did not train the extended fields or score Pluto, CTTC, or Qwen.
162:1. **Canonicalize the inputs.** Accept integer `codes[B,2]`, a task enum, `PhaseContext`, and optional M3 evidence. Compute \((u,v)\) and the task's output shift. The branch sees only canonical inputs, invariant context, and fixed-frame pilot evidence. Return exact uniform probabilities immediately for `q3_indep`, and for `q2` when there is no usable shared pilot.
176:### M4 interface — `PhaseContext` for Pluto and CTTC
```

### Event #65: `item.completed` — `command_execution` (`item_62`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "Get-Content -Encoding utf8 research/stage4_math_and_design/M6/THEOREM_M6.md | Select-Object -Skip 119 -First 12; rg -n 'quotient.separate|Zero' research/stage4_math_and_design/M1/THEOREM_M1.md | Select-Object -Last 12"
```

**Aggregated Output:**

```text
### 4.1 Four arms and host contribution

| Arm | Evidence path and score |
|---|---|
| 0 · Zero | Separate window, phase and reference slots, zero numeric phase values. Signed metadata stays in the shared permitted text. Score with Qwen. |
| 0b · Naive-Additive | Same scaffold with separate populated phase slots. Frozen nonlinear attention may mix them. The additive obstruction in §3 applies to centered additive logits and their fixed tangent analogue. |
| 1 · Host-Pure | Canonicalize using the task-specific action, couple cache-derived evidence with continuous phase and lag, and use fixed-length RefSufficientPool evidence. Project those statistics into the frozen host. Only the host's candidate scores determine the canonical prediction. Apply the final label permutation by integer indexing. |
| 2 · Physical+HostResidual | Independently fit the same coupled-token family. The prediction in canonical order is $\operatorname{softmax}(\log p_0+\delta)$, where $p_0$ is the same-information, dev-calibrated physical prior and $\delta=.15\,\Pi\tanh(\Pi s_{\rm host}/T_R)$. $T_R$ acts only on the host residual. |

For the four-arm comparison all old windows use the same **4-bit s17 per window**, with the same metadata, authorized references and fixed-shape reference plumbing. Arm 1 may use continuous likelihood features derived from that common information. Arm 1 never receives physical answer probabilities or logits. Arms 0 and 0b isolate numeric phase slots. Arms 0b and 1 compare the quotient/coupling package. Isolating the separate training effects of canonicalization and coupling would require a crossed training design. The 16-bit cache is a CPU ablation and export in this lock. If the s17 branch fails risk eligibility, any GPU upgrade to 16 bits requires a new protocol sharing that input across arms.

At residual zero, $\operatorname{softmax}(\log p_0)=p_0$ before the shared scoring floor. Each pairwise residual log-odds shift has magnitude $\le .30$. A wrong physical leader with a gap greater than .30 cannot be reversed. Expanding the normalizer also bounds each single-case NLL gain by .30. Measure Arm 2 versus physical-only and versus shuffled residual with **the same physical prior vector on each row**. Measure Arm 1's native score independently. M1's CPU hybrid reached 64.62% and tied the physical comparator. M2's retrospective blind-temperature audit reversed Stage 3D's G1 phase NLL lead. Fit calibration on disjoint dev folds, then freeze it. [M1 results](../M1/THEOREM_M1.md), [M2 forensic audit](../M2/THEOREM_M2.md)
308:- **Host-Pure uses the signed slots.** On G1, sign swap and balanced shuffle increase Arm 1's NLL by \(2.0172\) and \(1.0072\) nats. On G3 the increases are \(1.9229\) and \(0.9677\). The zero-slot retrain retains the public text and falls to 15.38% and 16.80% on G1 and G3. The quotient-separate control still passes 55% in both cells, so the numeric slot matters even though explicit multiplication is dispensable on this host.
313:For the next frozen-Qwen comparison, keep the quotient and the physical-only control fixed. Carry the quotient-separate comparator alongside Host-Pure so the effect of explicit multiplication can be measured. Treat Physical+HostResidual as an experimental increment on Arm 3 and require a held-out gain under the same dev calibration and fixed-text controls. M2's simultaneous gate, Qwen candidate geometry, 16-bit predictive gains, and hardware accuracy remain separate measurements. Intervals here are descriptive paired block intervals for the fixed three seeds. [Round 2 numerical receipt](results_m1.json) · [Handoff](reply_r2.md)
```

### Event #66: `item.completed` — `file_change` (`item_63`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M4\\verify_m4.py",
    "kind": "update"
  }
]
```

### Event #67: `item.completed` — `file_change` (`item_64`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M4\\THEOREM_M4.md",
    "kind": "update"
  }
]
```

### Event #68: `item.completed` — `file_change` (`item_65`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M4\\reply_r2.md",
    "kind": "update"
  }
]
```

### Event #69: `item.completed` — `file_change` (`item_67`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M4\\THEOREM_M4.md",
    "kind": "update"
  }
]
```

### Event #70: `item.completed` — `command_execution` (`item_66`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$env:UV_CACHE_DIR=(Resolve-Path '"'research/stage4_math_and_design/M4/.uv-m4-r2').Path; "'$env:TEMP=$env:UV_CACHE_DIR; $env:TMP=$env:UV_CACHE_DIR; $env:UV_OFFLINE='"'1'; "'$env:UV_NO_PROJECT='"'1'; "'$env:UV_LINK_MODE='"'copy'; "'$env:OPENBLAS_NUM_THREADS='"'1'; uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B research/stage4_math_and_design/M4/verify_m4.py --check-http"
```

**Aggregated Output:**

```text
Verified frozen s17 identity and pinned Python/NumPy/SciPy
Verified static splits, historical manifests and Modal arithmetic
PASS 168 mapping cells, 32 nuisance cells, 6 dual-cache profiles, 219 causal bundles, results_m4.json
```

### Event #71: `item.completed` — `command_execution` (`item_68`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$r=Get-Content -Encoding utf8 -Raw research/stage4_math_and_design/M4/results_m4.json | ConvertFrom-Json; $r.modal_budget | Select-Object m6_round2_ledger_matched_row_for_row,stage4_scheduled_without_retry_usd,stage4_scheduled_usd_cap,stage4_total_usd_cap,unallocated_reserve_below_hard_cap_usd,stage4_a100_seconds_cap | ConvertTo-Json -Compress; $r.modal_budget.pilot | ConvertTo-Json -Compress; $r.causal_frontend_contract.m3_pool_integration | ConvertTo-Json -Compress; $r.causal_frontend_contract | Select-Object byte_exact_refpool_fixture_bundles,rejected_forbidden_ref_attempts,probability_gauge_max_abs_defect,schur_v_joint_abs_log_likelihood_error | ConvertTo-Json -Compress; $r.dataset_manifests.live_http | ConvertTo-Json -Compress'
```

**Aggregated Output:**

```text
{"m6_round2_ledger_matched_row_for_row":true,"stage4_scheduled_without_retry_usd":"8.39064764","stage4_scheduled_usd_cap":"9.07079564","stage4_total_usd_cap":"9.50","unallocated_reserve_below_hard_cap_usd":"0.42920436","stage4_a100_seconds_cap":10987}
{"scope":"same single seed for all four arms","max_steps_per_arm":69,"max_total_steps":276,"previous_m6_pilot_steps":173,"previous_m6_optimizer_denominator":960,"previous_m6_pilot_usd":"0.49121800","planned_arm_denominator":384,"dev_rows_cap":4000,"futility_probability_floor":0.1,"pilot_including_smoke_usd_cap":"0.87663520","previous_planning_reference_z":-0.270438,"previous_planning_reference_information_fraction":0.18,"current_independent_proxy_cutoff":0.046796176916531124,"conditional_independent_blind_decision_only":true,"live_result":"pending"}
{"peer_wire_version":2,"tested_layouts":2,"max_posterior_log_error":3.5527136788005009E-15}
{"byte_exact_refpool_fixture_bundles":219,"rejected_forbidden_ref_attempts":1533,"probability_gauge_max_abs_defect":6.6613381477509392E-16,"schur_v_joint_abs_log_likelihood_error":0.0}
{"mendeley_files":{"metadata_reachable":false,"detail":"HTTP Error 403: Forbidden"},"zenodo_files":{"http_status":200,"metadata_reachable":true}}
```

### Event #72: `item.completed` — `command_execution` (`item_69`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n '8\\.84181248|6400|6,400|1,363|2\\.57549376|10684|10,684|\\.65818752' research/stage4_math_and_design/M4/THEOREM_M4.md research/stage4_math_and_design/M4/reply_r2.md research/stage4_math_and_design/M4/verify_m4.py; Resolve-Path -LiteralPath 'research/stage4_math_and_design/M4','research/stage4_math_and_design/M4/.uv-m4-r2'; Get-ChildItem -LiteralPath 'research/stage4_math_and_design/M4/.uv-m4-r2' -Recurse -Force -Attributes ReparsePoint | Select-Object -ExpandProperty FullName; git rev-parse HEAD"
```

**Aggregated Output:**

```text

387e7bc749150a69fb317b3f1351ad0d07b3bfad
Path                                                                                                       
----                                                                                                       
C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M4          
C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M4\.uv-m4-r2


```

### Event #73: `item.completed` — `file_change` (`item_70`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M4\\reply_r2.md",
    "kind": "update"
  }
]
```

### Event #74: `item.completed` — `command_execution` (`item_71`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "uv cache clean --cache-dir 'C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M4/.uv-m4-r2' --offline"
```

**Aggregated Output:**

```text
Clearing cache at: research/stage4_math_and_design/M4/.uv-m4-r2
Removed 4948 files (295.5MiB)
```

### Event #75: `item.completed` — `command_execution` (`item_72`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "Get-ChildItem -Force -Name research/stage4_math_and_design/M4; git rev-parse HEAD; rg -n '9\\.07079564|3\\.55e|M6.*rows' research/stage4_math_and_design/M4/reply_r2.md; Test-Path -LiteralPath 'research/stage4_math_and_design/M4/.uv-m4-r2'"
```

**Aggregated Output:**

```text
reply_r1.md
reply_r2.md
results_m4.json
THEOREM_M4.md
verify_m4.py
387e7bc749150a69fb317b3f1351ad0d07b3bfad
3:**The shared M4/M6 ledger schedules $9.07079564 including retry, under a $9.50 gross cap with a $0.00 net spend guard.** Standard A100-80GB Function pricing is `$0.000694 + 2×$0.0000131 + 16×$0.00000222 = $0.00075572/s`. Round 1 used 12 GiB, which accounts for `$0.00000888/s` of the difference. Stage 3D observed `52,972.5 MiB` training and `45,409.9 MiB` eval GPU peaks. All scheduled GPU work uses A100-80GB and requires fresh Stage 4 timing and parity receipts. A 24 GiB alternative needs a separately charged ≤$0.10 receipt proving ≤20 GiB peak and scorer parity before any blind rows. [Modal pricing](https://modal.com/pricing) · [Stage 3D measurements](../../stage3d_qwen_signed_phase/REPORT.md) · [M6 shared ledger](../M6/THEOREM_M6.md#44-the-sole-stage-4-modal-ledger)
10:The pinned `uv` verifier exited 0 on Python 3.12.7, NumPy 2.3.5 and SciPy 1.17.0. All 15 resource rows and the cost total matched M6. The M3 v2 posterior agreed with the full joint likelihood to `3.55e−15` in log probability. The maximum RX/TX probability-action defect was `6.66e−16`, 219 canonical projection tests were byte-exact, and 1,533 forbidden-reference attempts were rejected. The replay used generated IQ, archived geometry and the frozen `s17` weights. The Zenodo metadata probe returned HTTP 200. Mendeley returned HTTP 403, and its archived manifest verification passed. No Modal jobs were dispatched. [Verifier](verify_m4.py) · [Results](results_m4.json)
False
```

### Event #76: `item.completed` — `agent_message` (`item_73`)

M4 Round 2 is complete. The pinned `uv` verifier exited 0 and matched all 15 resource rows in M6’s Round 2 ledger.

- **Modal.** The shared A100-80GB plan schedules **$9.07079564 including retry**, under the **$9.50 gross cap** with a **$0.00 net spend guard**. The matched pilot costs **$0.87663520** for 69 steps per arm, setup, and 4,000 disjoint dev scores. It replaces the earlier **$0.49121800 single-candidate pilot**. Any continuation stays conditional on four connectors trained at the same fixed seed.
- **Dual cache.** The 16-bit word preserves the `s17` prefix and adds measured radial and residual information in one CPU pass. The registered window payload is at most **87,768 bytes**, with zero added GPU rows. The fitted 16-bit reader lowered joint symbol-and-variance NLL by **0.294–0.348 nats** across six synthetic profiles. At +20 dB, almost all of the gain came from variance-state inference.
- **Causal lock.** All **219** simulated split fixtures passed byte-exact checks, **1,533** forbidden reference attempts were rejected, and the maximum RX-gauge/TX-relabel probability defect was **\(6.66\times10^{-16}\)**. The actual M3 v2 pool passed both pilot layouts. All nine CTTC captures remain in the analysis, with `f708_snr10_QPSK` held out of fitting and calibration.

Real-radio timing and provenance, A100 throughput, and live credit qualification remain pending. Round 2 cloud spend was **$0.00**.

[Round 2 handoff](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M4/reply_r2.md) · [Protocol and ledger](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M4/THEOREM_M4.md) · [Verifier](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M4/verify_m4.py) · [Results](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M4/results_m4.json)



### Event #77: `turn.completed`

- Usage: `{"input_tokens": 23488905, "cached_input_tokens": 23180416, "cache_write_input_tokens": 0, "output_tokens": 111021, "reasoning_output_tokens": 25930}`


## Round 2 — Final Output (`reply_r2.md`)

```markdown
M4 Round 2 is complete. The pinned `uv` verifier exited 0 and matched all 15 resource rows in M6’s Round 2 ledger.

- **Modal.** The shared A100-80GB plan schedules **$9.07079564 including retry**, under the **$9.50 gross cap** with a **$0.00 net spend guard**. The matched pilot costs **$0.87663520** for 69 steps per arm, setup, and 4,000 disjoint dev scores. It replaces the earlier **$0.49121800 single-candidate pilot**. Any continuation stays conditional on four connectors trained at the same fixed seed.
- **Dual cache.** The 16-bit word preserves the `s17` prefix and adds measured radial and residual information in one CPU pass. The registered window payload is at most **87,768 bytes**, with zero added GPU rows. The fitted 16-bit reader lowered joint symbol-and-variance NLL by **0.294–0.348 nats** across six synthetic profiles. At +20 dB, almost all of the gain came from variance-state inference.
- **Causal lock.** All **219** simulated split fixtures passed byte-exact checks, **1,533** forbidden reference attempts were rejected, and the maximum RX-gauge/TX-relabel probability defect was **\(6.66\times10^{-16}\)**. The actual M3 v2 pool passed both pilot layouts. All nine CTTC captures remain in the analysis, with `f708_snr10_QPSK` held out of fitting and calibration.

Real-radio timing and provenance, A100 throughput, and live credit qualification remain pending. Round 2 cloud spend was **$0.00**.

[Round 2 handoff](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M4/reply_r2.md) · [Protocol and ledger](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M4/THEOREM_M4.md) · [Verifier](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M4/verify_m4.py) · [Results](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M4/results_m4.json)

```
