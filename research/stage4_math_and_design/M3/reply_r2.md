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
