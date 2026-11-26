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
