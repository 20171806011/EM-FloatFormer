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
