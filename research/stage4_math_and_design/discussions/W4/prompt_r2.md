# Track W4 — Round 2 Cross-Track Integration & Master Systems/Budget Lock

**Your Track:** `W4` (Systems/VRAM Equations, Mixed-Precision QLoRA, Cold-Start Anti-Shock Curriculum, Adapter Merging & Modal `$12.88` Decision Boundary)
**Persistent Thread:** Keep full continuity with your Round 1 derivations in [THEOREM_W4.md](../../weight_mod/W4/THEOREM_W4.md) and [verify_w4.py](../../weight_mod/W4/verify_w4.py).

---

## Verified Converged Findings & Cross-Track Reconciliation (`W1, W2, W3, M1–M6`)

Your Round-1 deliverables matched all 10 Stage 3D A100-80GB VRAM benchmarks within **`0.332%`** (`4,205,751,296` exact `Qwen3.5-4B` params, `|V| = 248,320`), proved `NF4` DeltaNet gate quantization fails `+20 dB L=64` by `+0.728 nats` while **`Mixed MLP-NF4 + DeltaNet-BF16`** drops KL to `0.00032 nats`, proved `3.60×` Cold-Start Gradient Shock (`9.73×` KL drift), verified `Weight-Space Soup` (`1.0089` NLL on unseen mixed capture), and locked the **`(0.030, 0.315]` nat** VoI gate inside `$12.88` (`Branch 1c: $11.24` all-retry total, `Billed Cost: $0.00`).

Now converge all four tracks into ONE unified Stage 4–6 systems & Modal ledger addressing these exact points from `W1, W2, W3, M4, M6`:

1. **Upgrade Phase-B Architecture from Generic `5.06M` LoRA to the Unified `W1 + W2 + W3` Stack (`<= 2.20M` Params!)**:
   - Notice your Round 1 modeled a generic `5.06M` BF16 LoRA (`86.9 MiB` static, `2.21 s/step`). Instead, `W1`, `W2`, and `W3` converged on a **far leaner, symmetry-locked, anti-forgetting surgical stack (`~1.01M–1.64M` adapter params `<= 2.20M` M1/M5 attribution cap on `Qwen3.5-4B`)**:
     - `M1 EquivariantEvidenceConnector` + Quotient Wrapper (`0.49M–0.69M`, bit-exact `0.000e+00` $C_4 \times C_4$ defect);
     - `W1`: 1-line zero-param **slow-head `dt_bias` warm-start** on dedicated EM GDN heads + Tier-1A/1B selective GDN (`in_proj_a/b/qkv/out` on `[11, 15, 19, 23]` + optional 2–4 early-layer slow-head preservation exception);
     - `W2`: **`FP32` Complex-Cayley (`O_proj`, `r_c = 4`, $8 \times 8$ complex Woodbury solve)** + **Schur-Irrep Dynamic Hyper-LoRA (`r = 3..11`)** on Full-Attn layers `[12, 16, 20, 24, 28]` (`~0.16M–0.33M`, `< 0.5%` FLOPs overhead; **NEVER 4-bit quantized** so $\|Q^\top Q - I\| \le 10^{-6}$ holds!);
     - `W3`: **Complex-Symmetrized Input Nullspace Projector $P_{\perp, C}^{\mathbb{C}}$** (`39.38 MiB` BF16 on 4B, `100.76 MiB` on 9B, `245.00 MiB` on 27B) + **Two-Tier Token-Position Gate ($m_{\text{EM}}(t)$)** + **Bayes-Posterior KL + Counterfactual Slot Margin (`CF-Margin` with paired `±ω` / slot-flip sub-batch)** + MixKL prose anchor!
   - **Systems Accounting Task 1 (Exact VRAM & Step-Time Recalibration)**:
     - Model the **exact** incremental VRAM and A100 step latency of BOTH:
       - **Sub-batch CF-Margin + MixKL Anchor** (`sub = 8` or `16` counterfactual/prose pairs) + `39.38 MiB` BF16 $U_k$ + FP32 Cayley $8 \times 8$ Woodbury solve (`t_step` relative to Stage 3D's `1,975.7 ms/step`), verifying peak VRAM stays well below M6's **`65,000 MiB`** gate (`~53.2–57.5 GiB`? Check exact numbers!).
       - **Phase-B Two Modes Trade-off (Answering `W3`)**: Quantify **(Mode I)** freezing Phase-A Layer-0 connector $E_{\theta_A}$ + stop-grad at Layer 10 (layers `1–10` skip backward completely — how many GiB & `ms/step` does stopping autograd at Layer 10 save on `Qwen3.5-4B / 9B / 27B`?) vs **(Mode II)** co-tuning $E_\theta$ at `0.15 η_A` (`53+ GiB` full 32L autograd, required ONLY if `W1`'s early-GDN slow-head warmgate touches layers `1–10`).

2. **Reconcile Phase-B Step Compression (`192 -> 69–96 Steps`) & Tier-P1 LEMMO Parity Inside `$12.88`**:
   - Both `W2` (`4c_quotient_cayley_schur_hyper_lora`: **`64.60% / 0.9526` already at Step 69!**) and `W3` (`Bayes-KL + CF-Margin`: `113.73×` lower gradient variance, **`61.26% ± 0.75 pp` at Step 69**, vs `44.30%` for 1-hot CE!) proved convergence happens **2×–3× faster than `1-hot CE` `384`-step runs**!
   - Compute the **exact Decimal Modal Ledger** (`A100-80GB 2 cores 16 GiB = $0.00075572/s`, anchored identically to **M4 §3.4 / M6 §4.4** `Branch 0: $8.3906 / $9.0708 w/ retry / $9.50 cap`):
     - If Phase B runs **`96` surgical steps** (instead of `192`, thanks to Step-69 W2+W3 convergence!) OR **`69` steps**, how much total USD (with retry!) does `Branch 1-Surgical-96` cost — and can the saved budget inside `$12.88` (`Billed Cost: $0.00`) simultaneously fund a **matched-access Tier-P1 `LEMMO-Matched-LoRA` (`<= 2.20M` params) comparison arm on A100-80GB**?!

3. **Canonical Basis Gauge-Free Merging Check (Resolving Your Prop 4.1 Question to `W2`)**:
   - In `verify_w4.py` Exp 2A, compare unconstrained static LoRA regime merging (`cos = -0.0908`, `Γ_12 = 2.268`, `1.0×` Task Arithmetic `1.0655` vs Linear Soup `1.0089`) side-by-side with **Canonical Schur-Irrep / Complex-Cayley Phase-B Merging**: test whether fixing W2's aligned basis eliminates `Γ_12` asymmetric interference and improves zero-retraining generalization on your unseen mixed Pluto+CFO+Spur capture!
   - Also explicitly reconcile the VoI upper cutoff $\Delta_{\text{route,max}} \in [0.25, 0.315]$ nats with `W1`'s Layer-24/Gap-64 $\sigma_3(D_{\text{answer}})$ bottleneck!

---

## Round 2 Action Items for Track W4 (Execute Strictly Inside `weight_mod/W4/`)

1. **Update `verify_w4.py` (`<= 90 s` CPU Wall-Clock) & `results_w4.json`**:
   - Retain all existing Round-1 Stage 3D VRAM calibration (`<= 0.332%`), Exp 1 Anti-Shock, Exp 2B NF4 vs Mixed MLP-NF4+DeltaNet-BF16, and Decimal budget checks.
   - **Add Round 2 checks**:
     1. Exact Unified `W1+W2+W3` (`<= 2.20M` params) A100-80GB VRAM & step-time breakdown for **(Mode I) Stop-Grad Layer-10 (`E_θA` frozen)** vs **(Mode II) Co-Tuned `0.15 η_A`** + Woodbury `8×8` Cayley + BF16 $U_k$ + CF-Margin sub-batch (`<= 65,000 MiB` verified across `4B / 9B / 27B`).
     2. Decimal Modal `$12.88` Ledger comparing **`Branch 0 ($9.0708/$9.50)`**, **`Branch 1-Surgical-96`**, **`Branch 1-Surgical-96 + Matched Tier-P1 LEMMO`**, and **`Branch 1c-192`**, with 10,000-trial Monte Carlo verification confirming **`max_cost <= $12.88` (`Billed Cost: $0.00`)** on `100%` of trials!
     3. Extended Exp 2A: Unconstrained LoRA merging vs **W2 Canonical Schur/Cayley Merging** (compare `Γ_12`, `cos_sim`, unseen mixed-capture Acc/NLL across Soup, TIES, Task Arithmetic).
2. **Update `THEOREM_W4.md`**:
   - Append clean Round 2 sections presenting the Unified `W1+W2+W3+W4` Systems & Modal Master Decision Ledger (`<= 2.20M` parity, Stop-Grad savings vs Co-Tuning, `Branch 1-Surgical-96 + Tier-P1 LEMMO` feasibility inside `$12.88`, canonical-basis merging results, and VoI link to W1's $\sigma_3(D_{\text{answer}})$).
3. **Verify & Report**:
   - Run `uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 --with torch python -B research/stage4_math_and_design/weight_mod/W4/verify_w4.py` and confirm **exit code `0`** and **`<= 90 s`**!
   - Return a concise summary of all updated Round 2 deliverables and exact numbers. Zero edits outside `weight_mod/W4/`.
