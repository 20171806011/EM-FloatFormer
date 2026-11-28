# Late-Query Cache Frontier: Four Conjectures & Lean 4 + Mathlib4 Formal Verification

**Date:** 2026-11-28  
**Branch:** `research/late-query-frontier-lean4` (branched from `research/late-query-cache-dimension`)  
**Toolchain:** `leanprover/lean4:v4.36.0` + `mathlib4` `v4.36.0` (exact config archived under `lean_env/`)

This directory extends the Phase 1 finite-guardrail baseline ([`../PROOF.md`](../PROOF.md)) across four orthogonal frontier directions of **late-query caching**, each independently researched, adversarially stress-tested, and machine-checked in Lean 4 + Mathlib4 with zero `sorryAx` and zero custom axioms.

---

## Summary of the Four Frontier Branches

| Branch | Conjecture & Scope | Theory Monograph | Lean 4 + Mathlib4 Module | Main Theorems | Kernel Audit (`verify.ps1`) | Key Results & Refinements |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| **1. WaveField** | Physical wave/EM evanescent-mode polylog law & near-field probe jump | [`WaveField/THEORY.md`](WaveField/THEORY.md) (65.7 KB) | [`WaveField/WavePolylog.lean`](WaveField/WavePolylog.lean) (1,120 lines) | **41** | `17.51s` · `CLEAN` | Two-regime crossover (`N_DOF(ka)` propagating → evanescent polylog); exact leading constants $\frac{m(\log_2(1/\varepsilon))^2}{8\log_2(R/a)}$ (2D: $m=2 \Rightarrow \frac{1}{4}$) & $\frac{m_0(\log_2(1/\varepsilon))^3}{24(\log_2(R/a))^2}$ (3D real Maxwell TE+TM: $m_0=2 \Rightarrow \frac{1}{12}$); near-field jump factor $(\log_2(R_1/a)/\log_2(R_2/a))^{d-1}$; Mie WGM resonance & heavy-tail breakdown bounds. |
| **2. CompInflation** | Computation-limited readout dimension inflation & shallow ReLU collapse boundary | [`CompInflation/THEORY.md`](CompInflation/THEORY.md) (64.3 KB) | [`CompInflation/CompInflation.lean`](CompInflation/CompInflation.lean) (854 lines) | **8** | `21.38s` · `CLEAN` | Linear readout rank obstruction $r \ge \dim\overline{\mathrm{span}}_{L^2}\{T_q\}$ (Ky Fan + Gram min-eigenvalue in general real `InnerProductSpace`, Vandermonde monomials); sharp regime split between joint VQ ($1/2$ slope) vs. per-coordinate scalar quantization ($K/2$ slope); Rademacher double-counting Lipschitz covering bound; depth-2 width-$W$ ReLU $L^\infty$ quadratic floor $\ge \frac{1}{8(W+1)^2}$. |
| **3. CausalPremium** | Observational vs. interventional causal cache premium $\Delta_d^{\mathrm{causal}}$ | [`CausalPremium/THEORY.md`](CausalPremium/THEORY.md) (48.2 KB) | [`CausalPremium/CausalPremium.lean`](CausalPremium/CausalPremium.lean) (998 lines) | **42** | `25.55s` · `CLEAN` | Exact separation of counterfactual $f_a^{\mathrm{cf}}$ vs. pre-treatment interventional $f_a^{\mathrm{do}}$; multi-node linear SCM backdoor intersection law $\Delta_d^{\mathrm{causal}} = \dim(\operatorname{im}\Theta_{YA}\cap\operatorname{im}\Theta_{YZ})$; canonical 2D bi-Lipschitz equivalence; local Fréchet Jacobian obstruction (`HasFDerivAt`); Y-graph/$S^1$/singularity counterexamples & 5 single-probe failure modes. |
| **4. MemoryPhase** | Sequential re-compression forgetting phase transition & causal-state critical dimension | [`MemoryPhase/THEORY.md`](MemoryPhase/THEORY.md) (76.6 KB) | [`MemoryPhase/MemoryPhase.lean`](MemoryPhase/MemoryPhase.lean) (1,097 lines) | **21** (+8 witnesses) | `14.88s` · `CLEAN` | Sharp phase transition at $d_{\mathrm{mem}}^* = \dim E^{u/c}$ & $b^* = H_{\mathrm{top}}(A)$; exponential blow-up ($|\lambda|>1$), Jordan-block polynomial divergence ($|\lambda|=1$), $|\lambda|=1$ zooming-coder vs. fixed-register split; super-critical uniform $(t,\tau)$ boundedness; Dobrushin/Doeblin multi-step SDPI contraction; periodic external retrieval $b_{\mathrm{eff}} = b + B_{\mathrm{ext}}/T$. |
| **Total** | **4 Orthogonal Frontier Branches** | **254.9 KB** | **198.2 KB · 4,069 lines** | **112** | **4 / 4 VERIFIED** | **All 112 `#print axioms` depend solely on `[propext, Classical.choice, Quot.sound]`** |

---

## Independent Lean 4 Kernel Audit & Reproducibility

Every `.lean` file in this directory was independently re-verified via `lean_env/verify.ps1` against `leanprover/lean4:v4.36.0` + `mathlib4` `v4.36.0` with strict static anti-cheat checks (no UTF-8 BOM, no `sorry`, `admit`, `native_decide`, `implemented_by`, or custom `axiom` declarations, and narrow Mathlib imports only):

```powershell
powershell -ExecutionPolicy Bypass -File C:\lean4\math_lab\verify.ps1 MathLab\Projects\LateQueryFrontier\WaveField\WavePolylog.lean
powershell -ExecutionPolicy Bypass -File C:\lean4\math_lab\verify.ps1 MathLab\Projects\LateQueryFrontier\CompInflation\CompInflation.lean
powershell -ExecutionPolicy Bypass -File C:\lean4\math_lab\verify.ps1 MathLab\Projects\LateQueryFrontier\CausalPremium\CausalPremium.lean
powershell -ExecutionPolicy Bypass -File C:\lean4\math_lab\verify.ps1 MathLab\Projects\LateQueryFrontier\MemoryPhase\MemoryPhase.lean
```

Each monograph (`THEORY.md`) includes an explicit **Scope Boundary** section mapping every machine-checked Lean theorem to its analytic/measure-theoretic counterpart without overstating formalization coverage.
