# CPU experiment on known frequency offset and third-window prediction

On the pinned Stage 3 encoder, using the correct signed frequency offset improved third-window sector accuracy by 43.85 percentage points at the published +0.35 rad/step setting. The paired 95% bootstrap interval was [42.64, 45.06] percentage points. This experiment changes the transport calculation in the direct readout. It supplies a controlled prediction for a future language-host intervention.

## Inputs and target

The run used the published simulator, NumPy encoder implementation, signed continuous-drift direct readout and frozen `equivariant_b8_s17.npz` from commit `69fbd29acdb4eba7961d56176693786b6181477a`. The held-out target was the received quarter-turn sector of the noisy third window under `repeat_x1`, before any pilot reveal. The two window codes together occupy 8 bits. The SNR was 2 dB, window gaps were 2 and 2 steps, and the per-step Brownian phase standard deviation was 0.10 rad. Each generator run used the upstream balanced blocks of 64 mothers. The sweep changes only the public angular rate.

The 128 published test mothers replayed at 60.15625% accuracy and 0.9729469191350633 nats for the signed direct readout. The same mothers gave 19.53125% when the offset was ignored and 3.90625% when its sign was reversed. The published host score on these mothers and their rotations was 20.57%. The aggregate resemblance to the offset-ignored score motivates a paired host intervention. All estimates in the table below use fresh seeds.

## Results

| Angular rate (rad/step) | Total deterministic rotation (rad) | Signed accuracy | Ignoring offset | Reversing sign | Signed NLL | Offset-ignored NLL |
|---:|---:|---:|---:|---:|---:|---:|
| −0.35 | −1.40 | 63.16% | 20.03% | 3.75% | 0.9563 | 1.8263 |
| −0.12 | −0.48 | 55.99% | 55.99% | 55.99% | 1.0119 | 1.1026 |
| 0.00 | 0.00 | 64.34% | 64.34% | 64.34% | 0.9465 | 0.9465 |
| +0.12 | +0.48 | 55.10% | 55.10% | 55.10% | 1.0215 | 1.1199 |
| +0.35 | +1.40 | 63.26% | 19.41% | 3.86% | 0.9564 | 1.8450 |

Each rate used 16,384 test mothers from four new seeds. Within each rate the three methods saw identical observations and cache codes. Across rates the generator reused symbols, Brownian innovations and receiver noise by seed. At zero offset all three probability calculations coincide. At both small nonzero offsets signed transport improved NLL while leaving the most probable sector unchanged on these data. At +0.35 the mean NLL difference (signed minus offset-ignored) was −0.88861 nats, with paired interval [−0.90497, −0.87214]. Both accuracy and a proper scoring rule therefore matter.

## Does the second cached window help in this particular regime?

The Stage 3 direct readout for pre-pilot Q3 uses the first code's stored four-symbol table and marginalizes over the future noisy phase. Continuous drift can in principle make the second code informative about the latent fractional rotation, since the second and third windows share part of their phase trajectory. We also estimated two frequency-specific discrete posterior tables from 131,072 additional simulator mothers per rate. One table conditioned on both codes, the other on the first code. Calibration and evaluation seeds were disjoint. Both tables had access to the fixed evaluation condition through independent calibration simulations, so they are diagnostics of cached information under that known generative model.

For each 4-bit code the frozen encoder returns a quadrant and one of four within-quadrant labels. Rotating the first received window and repeated third window together permutes their labels exactly. A separate quarter turn of the second window preserves the law by uniformity of its independent sent symbol. These group actions allow count pooling over rotations. The pair table estimates the sector relative to the first quadrant from the two within-quadrant labels. Each four-outcome cell received one-half pseudocount per outcome. The least occupied pair cell across the full sweep contained 944 calibration mothers.

At +0.35 the two-code and one-code tables both reached 63.26% accuracy. Their paired mean NLL difference (two minus one) was +0.000003 nats, interval [−0.000292, +0.000288]. Across all five settings their top-1 scores matched and all five paired NLL intervals included zero. The intervals resample the test data conditional on these fitted tables. Finer effects, calibration uncertainty and other phase laws remain open. These results support prioritizing signed transport in a host intervention at this condition.

## Replay and interpretation

From the repository checkout containing this package and its Stage 3 parent:

```sh
python path/to/D/check_d.py --repo . --out d_replay.json
```

The script requires NumPy and SciPy. It checks the published weight SHA-256 and original 128-mother result before evaluating the sweep. `results.json` records seeds, unrounded metrics and 2,000-draw paired block-bootstrap intervals stratified by generator seed. Published observations serve only the initial replay. The language host and its adapter were never invoked in this experiment. Similar accuracy after ignoring CFO and in the published host cannot by itself identify what the host computed. Testing that mechanism requires paired host predictions with the same cache and controlled offset representation.

[Upstream simulator](https://github.com/20171806011/EM-FloatFormer/blob/69fbd29acdb4eba7961d56176693786b6181477a/research/stage3_ministral_cache/data.py) · [Direct readout](https://github.com/20171806011/EM-FloatFormer/blob/69fbd29acdb4eba7961d56176693786b6181477a/research/stage3_ministral_cache/analytic.py) · [Encoder](https://github.com/20171806011/EM-FloatFormer/blob/69fbd29acdb4eba7961d56176693786b6181477a/research/matched_cache/learned.py) · [Weight](https://github.com/20171806011/EM-FloatFormer/blob/69fbd29acdb4eba7961d56176693786b6181477a/research/matched_cache/weights/equivariant_b8_s17.npz)
