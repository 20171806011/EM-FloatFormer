# Track M2 | Round 1 reply

**G1's locked phase NLL lead is entirely explained by differential scalar-temperature mismatch along the audited decomposition.** The paired \(L_{\rm zero}-L_{\rm phase}\) is \(+0.03565000\) nats. Subtracting each arm's in-cell temperature regret changes it to \(−0.00089337\). The exact ordered split is \(0\) shared Bayes term, \(+0.03654337\) temperature, \(+0.00012375\) projection at the fitted temperatures, \(+0.00000005\) prior KL and \(−0.00101717\) reverse-conditional KL. Locked G1 phase/zero score 23.80%/19.41% accuracy and 1.445229/1.480879 NLL. Both are below uniform in accuracy and above it in loss. S17 direct scores 64.62% and 0.941126. The blind-fitted NLLs are 1.385472 and 1.384578, with ECE 0.025608 and 0.051229.

| Q3 cell | Locked zero-minus-phase NLL | After seedwise \(T^*\) | Independent physical sign MI |
|---|---:|---:|---:|
| G1 | +0.035650 | −0.000893 | 0.363112 |
| G2 | −0.027760 | −0.027716 | 0.176866 |
| G3 | +0.037083 | −0.000430 | 0.344010 |
| G4 | −0.024917 | −0.026351 | 0.179398 |
| H | +0.018090 | −0.003695 | 0.350032 |

Both Q3 arms observe the same signed metadata and codes. Bayes sign MI is available to both. On G2 and G4 zero retains its advantage under optimal scalar scaling, with \(0.031042\) and \(0.030809\) lower projected reverse-conditional KL. Work F's seed-mean orientation gain is 0.009820 for phase and 0.005048 for zero. Zero retains signed text. On G1 phase, holding the other path matched, flipping text costs 0.003231 nats and flipping slots costs 0.017240, with slot effects \(−0.000641,+0.035214,+0.017147\) across seeds. The within-checkpoint slot response does not establish a stable gain across retrains. On Q2 valid, recalibration reduces the phase lead from 0.021807 to 0.001594. The fresh-resample lead reverses from +0.000757 to −0.002672. All seven Q2 controls are in the full report.

**Geometric pooling preserves the failure pattern.** G1 arithmetic/geometric give phase 17.05%/16.36% at NLL 1.411312/1.411586 and zero 16.43%/16.60% at 1.439122/1.445854. G3 also loses accuracy under both pools. G2 and G4 improve accuracy and NLL with both. On complete orbits the exact arithmetic loss gain is label-weighted Jensen. For G1 zero it is 0.04120584 versus 0.03488442 for the requested reverse-KL expression. The latter uses the pooled forecast as the label law.

**Stage 4 gate.** Require absolute accuracy/NLL against uniform, dev prior and matched zero, a preregistered random-label test, and a positive linearized NIG contrast with proposed \(\eta_{\min}=0.10\). Require dev-only ECE below 0.05, paired Brier improvement of at least 0.005 against the strongest locked baseline, and maximum equivariance/idempotence error at most \(10^{-5}\). Require all endpoints within each claim and apply Holm across separate synthetic, Pluto and CTTC claims. Add a fixed-text slot intervention for claims about signed-slot use.

- **M1** should cross accumulated angle and frequency, with matched text, slots, code, waveform and physical references in the intervention design.
- **M3** should compare native forecasts with arithmetic, geometric and representative readouts on common development data and lock calibration before blind evaluation. G1 fails the absolute gate under the raw and projected scores.
- **M4** should retain 131 synthetic blocks pending full-gate pilots. With G1 variance and 3 pp/0.04 nats true excess above the margins, the historical pair needs 7/5 independent blocks for 95% marginal power conditional on the three fixed seeds. Across retrains, three seeds cap even known-variance power at 71.54%/85.94%, regardless of block count. Ten paired seeds give proxy minima of 15/7 independent blocks or captures, or 7/5 CTTC sessions of eight clusters at proxy session ICC 0.10. At the observed G1 excess, the fixed-seed capture proxy needs 10/17. Treat Pluto's 16 captures and CTTC's nine sessions as separate design populations, measure their variance on independent pilots, and power every gate endpoint before locking real-data enrollment.

The pinned CPU run exited 0. The in-scope Stage 3D summary replay has maximum absolute difference 0. [Theorem and full results](THEOREM_M2.md) · [Verifier](verify_m2.py) · [Numerical results](results_m2.json)

