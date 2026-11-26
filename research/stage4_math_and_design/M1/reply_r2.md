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
