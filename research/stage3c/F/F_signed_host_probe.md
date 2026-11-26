# Can a frozen host use the sign of a carrier offset?

This study tests **output behavior** for Q3-repeat, two frozen four-bit window indices, no pilot, and an explicit public signed carrier rate. The earlier positive-CFO host accuracy does not identify sign blindness. Here two quite different rules make precisely the same hard decisions on every test case. Their responses to controlled metadata swaps differ.

## One physical intervention and one input intervention

Use the Stage 3 generator with the original C4 timing (h=2, ell=2), 2 dB, phase diffusion 0.10 rad per square-root step. For each mother reuse source symbols, initial rotation, both Brownian innovations and all additive complex noises. Change the physical drift between −0.35 and +0.35 rad/step and regenerate windows and labels. Each sign therefore has the correct physical marginal. For Q3-repeat the label is the **received sector of the newly simulated noisy third window**. For Q2 the transmitted X1 would be unchanged; its valid pilot would generally need regeneration. No Q2 results are claimed here.

Let z_s and y_s be the actual two-window cache and third-window target at physical sign s. Let H_t(z) be four ordered candidate probabilities under told sign t. Evaluate all four vectors P_st=H_t(z_s). Within a row, the two input caches, labels, timing and prompts (except signed rate) are identical. Swapping t while holding s fixed is a metadata intervention within a recorded sample. The off-diagonal is deliberately inconsistent metadata, not a second physical truth. Across rows both the observed cache and target can change. Equating y_- and y_+ would contaminate 93.92% of the paired cases here; 93.14% have different cache pairs.

For a full-input interaction test use the vector-valued balanced factorial expansion P_st=μ+sW+tM+stI, s,t in {−1,+1}. Its coefficients are respectively one fourth of the unsigned sum, the sum weighted by s, by t, and by st. These are exact algebraic identities per mother. The JSON reports E‖W‖², E‖M‖² and E‖I‖² separately to avoid response cancellation across mothers. W measures sensitivity to the changed observed waveform at fixed metadata. I quantifies how the metadata response differs between the mother's two physically generated caches. The coefficients alone do not say whether the changes follow physics.

## Direction, magnitude and symmetry on common cache support

Independently estimate the two reference tables R_t(z) on 65,536 calibration mothers per sign using the frozen s17 encoder. Pool certified quarter-turn orbits and add 0.5 pseudocount per bin. No reflection symmetry is assumed. A different item in the cache can represent the conjugate state: quarter-turn equivariance by itself does not imply a sign-reflection symmetry. Both signs are fitted from their own actual observations. This reference estimates a conditional distribution and has Monte Carlo error.

Enumerate all 256 cache pairs. Weight each by the smaller empirical sign-specific mass and renormalize over common support. The empirical overlap summed to 0.9622. Let D(z)=R_+(z)−R_−(z), B(z)=H_+(z)−H_−(z), and take the inner product summed over sectors and weighted over cache pairs. For nonzero ‖D‖ report three quantities:

1. Orientation gain a=⟨B,D⟩/‖D‖². This is the unique least-squares coefficient of B on D, obtained by differentiating ‖B−aD‖².
2. Residual magnitude ‖B−aD‖/‖D‖. Pythagoras gives the sum of squared aligned and residual components. An insensitive model and one responding orthogonally can both have a=0.
3. Mean sign TV, the weighted sum of ‖B(z)‖₁/2. It vanishes exactly when both outputs agree on every positive-weight cache. This is behavioral invariance on this particular sign pair and protocol. A near-zero test needs a predeclared numerical tolerance and independent replication for approximate tables or stochastic scorers. At zero or unresolved reference contrast withhold directional interpretation. Uniform posteriors and exact ties receive fractional credit across maximizers; near ties warrant reporting NLL and TV alongside hit rate.

Also rotate W1 and the repeated W3 together by one quarter-turn, keeping W2 fixed. In the frozen code, z1 moves to (z1+4) mod 16. Compare that host output with a cyclic shift of its four original sector probabilities. This detects broken reference-frame covariance, including some absolute-sector biases. A failed covariance check admits other explanations. A zero directional residual does not certify correct even components, likelihood calibration or accuracy. No combination of these black-box measurements identifies internal circuits or proves that an arbitrary host implemented a particular algebraic operation.

## Executed controls

Each physical comparison uses four fresh test seeds, 16 independent 64-mother blocks per seed, 4,096 mothers in total. Bootstrap intervals resample blocks within seeds and keep the entire physical/told-sign factorial together. They condition on the fitted reference. The logged NLL clips probabilities at 1e-12. All controls below have positive predicted mass, without clipping in this run.

| CPU decision rule | Orientation gain | Mean sign TV | Quarter-turn TV | Matched accuracy | NLL, swapped minus matched |
|---|---:|---:|---:|---:|---:|
| Calibrated signed reference | 1.00 | 0.679 | 0.000 | 62.77% | +1.960 |
| Mean of the two sign tables, told sign ignored | 0.00 | 0.000 | 0.000 | 32.96% | 0.000 |
| Always use the opposite table | −1.00 | 0.679 | 0.000 | 3.66% | −1.960 |
| Fixed sector bias (.4,.3,.2,.1) | 0.00 | 0.000 | 0.300 | 24.54% | 0.000 |
| 92% fixed bias + 8% signed table | 0.08 | 0.054 | 0.276 | 24.54% | +0.146 |
| Orthogonal metadata response | ≈0.00 | 0.300 | 0.299 | 24.23% | −0.016 |
| Uniform probabilities | 0.00 | 0.000 | 0.000 | 25.00% | 0.000 |

At 92% bias the greatest possible label-to-label perturbation is 0.08, smaller than 0.92 times the smallest original gap between sector 0 and a competitor (0.10). Hence the two bias rules always output hard sector 0 despite different probability responses. The orthogonal control is constructed by removing the projection onto D of a fixed contrast and taking a small symmetric perturbation about uniform. Its normalized residual is 0.482. The signed, ignored and reversed normalized gains are algebraic control checks. The independent physical test gives a signed reference swapped-minus-matched gain of **1.9599 nats**, 95% paired interval **[1.9219, 1.9995]**.

## Running an actual host

Run `OPENBLAS_NUM_THREADS=1 python check_f.py --repo /path/to/EM-FloatFormer --out results.json` against the inputs inherited from commit `cbf3d2c0`. Source generation and symmetry pooling are linked to the [simulator](https://github.com/20171806011/EM-FloatFormer/blob/69fbd29acdb4eba7961d56176693786b6181477a/research/stage3_ministral_cache/data.py) and [signed comparison](https://github.com/20171806011/EM-FloatFormer/blob/cbf3d2c0/research/stage3b_math/D/D_experiment.md) in the prior stage (`research/stage3b_math/D`).

To use a real frozen host, enumerate 256 pairs (z1,z2) at each told sign. Hold prompt template, condition identifier, question and candidate order, connector, model, scorer and locked temperature fixed; print the signed coefficient with enough precision to distinguish signs. If the host has both text and phase-vector inputs, change the sign of **both** consistently. Run single-modality contradictory interventions separately. Only the metadata may vary across told signs. Store normalized finite, nonnegative four-candidate probabilities in this JSON schema:

```json
{"task":"Q3_repeat","encoder_sha":"e67d9fbafb90791ab19549de7789798d818286d7f52d7ac566cc39df71e78356","rate_magnitudes":[0.35],"h":2,"ell":2,"sigma_phi":0.1,"snr_db":2.0,"sign_order":[-1,1],"pilot":"none","cache_order":"16*z1+z2","probabilities":"replace this string with an array of shape 2 by 256 by 4"}
```

Keep model, connector and prompt hashes plus temperature with the exported array. Supply `--host host_predictions.json` to compute the same metrics, preserving the unrevealed test targets from the simulator. A deterministic host seeing only these caches and metadata needs 512 four-candidate score calls; model input must contain no mother IDs, labels, test residuals or raw IQ. A stochastic decoder needs repeated calls at each cache/sign. C4 extrapolates beyond the original connectors' zero-rate training support, and these measurements apply to the fixed magnitude and condition. Extending this experiment to other magnitudes requires signed reference calibration at the new rate. No new LLM responses were used to compute the table above.
