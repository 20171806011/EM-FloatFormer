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
