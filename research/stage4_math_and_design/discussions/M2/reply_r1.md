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

