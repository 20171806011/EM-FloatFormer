# Phase 8 Manifold (`p8_manifold`) Prospective Study Protocol

> **Registration status**: Within-session prospective freeze recorded to `p8_manifold/results/freeze.json` prior to formal held-out training/evaluation (not an external public preregistration). Namespace `p8_manifold/` is strictly disjoint from prior phases `p4..p7` and from the separate classical multi-frame branches `p8/` and `p8c/`.

## 1. Scientific Hypotheses & Architecture Under Test

Phase 7 (`4dab1ac`) established that multi-hypothesis $\mathbb{C}^*$-equivariant uncertainty routing fixes catastrophic calibration collapse under missing pilots, but left four open bottlenecks:
1. **Attention vs LocalConv deadlock (`EM7-Axial == EM7-Local`)**: caused jointly by the hard-coded 3-ray `(0, 2, 4)` analytical projection and unsigned/untransported complex attention (`|q^H k|`, `weights @ v`).
2. **Known 3-ray support dependency**: reliance on fixed integer delays `(0, 2, 4)` and known erasure masks.
3. **Sequential Pass 1 -> Pass 2 restart regression**: re-running Pass 2 from scratch degraded BLER (`-2` / `-64` blocks).
4. **In-branch epistemic variance & capacity scaling**: unmeasured capacity scaling across parameter orders of magnitude.

`p8_manifold` introduces a dictionary-free Off-Grid multi-path crucible (`tau in [0, 6.5]`, variable `P in {2, 3, 4, 5}`, linear-chirp Doppler `nu in [0.015, 0.048]`, per-burst randomized 64-pilot masks, `SNR = 15.0 dB`, `20 x 32` OFDM with `CP = 16`, 18 LDPC `(128, 64)` codewords = 576 16-QAM payload symbols per burst) and three original constructions:
- **Construction A (`CoherentTransportLayer`, `variant="transport"`)**: Strictly 1-homogeneous $\mathbb{C}^*$-equivariant signed Hermitian inner product $\operatorname{Re}((q')^\dagger k')$ with factorized $U(1)$ unitary parallel transport $e^{+i\langle \omega, z_i\rangle} \sum_j \alpha_{ij} (e^{-i\langle \omega, z_j\rangle} v_j)$ preserving raw 1-homogeneous pilot amplitudes in `V` across 1-hop 2D Pilot-to-Grid (`640 x 64`) + 2D Separable Grid (`Freq -> Time`).
- **Construction B (`CrossHypothesisJeffreysLayer`)**: Exact $\mathbb{C}^*$-invariant pairwise Jeffreys divergence $D_J(p_m \| p_{m'})$ driving `M x M` cross-hypothesis interaction and Jeffreys-weighted branch evidence routing.
- **Construction C (`EM8ManifoldReceiver`)**: Dictionary-free 4-chart strictly PSD periodic chordal $S^1$ 2D GP manifold initialization, blind redescending + Student-$t$ observation reliability inference, banded-ICI (`B_ici = 2`, 5 diagonals) epistemic-aware Gaussian cavity demapping, exact 3-term predictive variance decomposition (`aleatoric + epistemic + between`), and zero-restart exact GP posterior Schur-complement sequential assimilation (`assimilate()`).

## 2. Benchmark Conditions (`OFFGRID_CONDITIONS`)

| Condition | Pilot Topology (64 TX pilots/burst) | Corruption / Holdout Protocol |
| :--- | :--- | :--- |
| `O0_random_sparse` | Per-burst uniform random 64-pilot subset on `20 x 32` | Clean observed pilots |
| `O1_roving_island` | Two random edge bands (48 pilots) + roving `2 x 8` interior island (16 pilots) | Wide interior gap testing 1-hop non-local reach |
| `O2_blind_burst_erasure` | Per-burst random 64 pilots | 8 unknown erasures (`y=0`) + 8 unknown `8x..12x` impulses (`valid_mask=True`, `blind=True`) |
| `O3_holdout_sequential` | Same topology as `O1_roving_island` | 8 of 16 roving-island pilots strictly masked out in Pass 1 (`probe_mask`, zero leakage) and assimilated in Pass 2 via `assimilate()` |

## 3. Exact Comparator Arms (`FORMAL_ARMS`)

> **Scaling-Ladder Capacity Selection (Pre-Formal)**: Prior to running any formal seed (`802611190000`), the disjoint 3-rung Scaling-Law Ladder (`p8_manifold/results/scaling_ladder.json`, seeds `89000000`, `40` steps, `lr=0.0035`) revealed a sharp fixed-budget Chinchilla capacity optimum: `Small (2,063 params)` strongly outperformed `Medium (12,847 params)` and `Large (40,003 params)` across all four conditions (`O0` BLER `57.29%` vs `68.11%` vs `86.05%`; `O1` `82.18%` vs `86.86%` vs `94.91%`; `O3` `79.34%` vs `84.90%` vs `93.75%`), while `EM8-Transport` maintained a widening BCE/robustness advantage over `EM8-Local` across all three rungs (`+0.0034 -> +0.0096 -> +0.0501` BCE on `O1`). Consequently, the prospective formal `N = 10` study locks at the Chinchilla-optimal `Small` architecture (`d_model=8, n_layers=1, n_heads=2, steps=60`, exact `2,063` real params parity).

1. **`EM8-Transport`** (`2,063` real params; `d_model=8, n_layers=1, n_heads=2, M=4, em_steps=2`) — Full Construction A (multi-scale geodesic $U(1)$ parallel-transport wave attention) + B + C.
2. **`EM8-AbsNoTransport`** (`2,063` real params — exact integer parity & step-0 output identity) — Same global topology, geodesic bias, and parameters, ablated to unsigned `|q^H k|` + real cosine coordinate bias + untransported `weights @ v`.
3. **`EM8-Local`** (`2,063` real params — exact integer parity & step-0 output identity) — Exact same parameter tensors restricted strictly to `3 x 3` local coherent neighborhood (zero remote reach).
4. **`EM8-SingleHyp`** (`M=1`, `transport` backbone) — Isolates multi-chart uncertainty and cross-hypothesis Jeffreys routing.
5. **`EM8-Fixed`** (`0` learned params, `M=4`) — Analytical GP multi-chart + blind Student-$t$ + cavity + Schur-GP assimilation control.
6. **`OffGrid-2D-LMMSE`** (`0` learned params) — Fair dictionary-free classical 2D GP/Wiener LMMSE comparator.

## 4. Prospective Formal Sample Size, Seeds & Pre-Registered Inference Families

- **Base commit**: `4dab1ac9ee15b4f1a7f78ae05486c55a2dc3bd72` (inherited files strictly unmodified).
- **Seed base**: `SEED_BASE = 802611190000` (disjoint from Dev `88000000` and Scaling Ladder `89000000`).
- **Pairs**: `N = 10` fresh independent pairs (`pair_idx = 1..10`), each evaluated over `4` batches × `B = 8` bursts × `18` LDPC blocks = `576` blocks/pair (**`5,760` LDPC blocks per condition**).
- **Training recipe**: `60` Adam steps (`B = 8`, `lr = 0.0035` warm-cosine with `5`-step warmup to `0.1x`, cycling `O0..O3`, `lambda_hold = 0.08`, `lambda_nmse = 0.01`, `lambda_ici = 0.05`, `lambda_cal = 0.02`, `anchor_l2 = 1e-4`, `grad_clip = 1.0`, `probe_weight = 0.10`, `float64/complex128`, `torch.set_num_threads(1)`, `torch.use_deterministic_algorithms(True)`).
- **Statistical inference**: Exact paired two-sided / directional sign-flip permutation tests over all $2^{10} = 1,024$ sign assignments (`p7.statistics.exact_sign_test`) with within-family Holm-Bonferroni correction across 4 pre-registered families:
  - **Family 1 (Non-Local Transport Attention vs Parameter-Matched LocalConv, 4 tests)**: `EM8-Local - EM8-Transport` on `O1` BLER, `O1` BCE, `O0` BLER, `O0` BCE.
  - **Family 2 (Construction A Signed Unitary Transport vs Unsigned Untransported, 4 tests)**: `EM8-AbsNoTransport - EM8-Transport` on `O1` BLER, `O1` BCE, `O0` BLER, `O0` BCE.
  - **Family 3 (Blind Defense & Multi-Chart Calibration, 3 tests)**: `O2` BLER (`OffGrid-2D-LMMSE - EM8-Transport`), `O2` BCE (`EM8-SingleHyp - EM8-Transport`), `O1` BCE (`EM8-SingleHyp - EM8-Transport`).
  - **Family 4 (Held-Out Density & True Bayesian Sequential Assimilation on `O3`, 4 tests)**: Holdout NLL (`EM8-SingleHyp - EM8-Transport`), Pass 1 -> Pass 2 BLER gain (`> 0`), Pass 1 -> Pass 2 BCE gain (`> 0`), and uncertainty-selective revision (`top25_prob_shift - bot25_prob_shift > 0`) on `EM8-Transport`.

## 5. Independent Audits & Reproducibility Gates

1. **Exact Checkpoint Replay (`p8_manifold/replay.py`)**: All `40/40` formal checkpoints verified for SHA-256 integrity, `0` block/bit diff, and `< 1e-12` BCE/NLL diff on Batch 0 across `O0..O3`.
2. **Formal $\mathbb{C}^*$ Gauge Equivariance Audit (`p8_manifold/audit_gauge.py`)**: All `40` trained checkpoints × `3` conditions (`O1, O2, O3`) × `27` `SCALES` (`1e-20..1e15`) = `3,240` complex transforms verifying `0` coded flips, `all_finite`, and `< 1e-12` relative equivariance (reporting both strict absolute and relative LLR diffs verbatim).
3. **Isolated 1-Thread CPU Latency (`p8_manifold/latency.py`)**: `2` warmups + `10` interleaved rounds for `B in {1, 8}` on `O0` and `O1`.
4. **Controlled 3-Rung Scaling-Law Ladder (`p8_manifold/scaling_ladder.py`)**: Separate disjoint seeds (`89000000`), `Small (2,059)` -> `Medium (12,831)` -> `Large (39,979)` comparing `EM8-Transport` vs `EM8-Local`.
