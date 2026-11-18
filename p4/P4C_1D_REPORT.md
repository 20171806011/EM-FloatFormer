# P4-C exploratory one-dimensional receiver study

On three paired development seeds, gauge normalization yielded the largest average reduction in uncoded bit error rate when the receiver gain varied. At unit gain its average benefit was substantially smaller. The joint pilot/data switch produced a small mixed effect. These results concern the one-dimensional P4-A receiver under a short training budget. The separate two-dimensional coded P4-D study uses a different endpoint.

## Matched factorial

All eight arms used two heads and one layer. Widths of 6, 8 or 10 kept the trainable parameter count between 3,122 and 3,342. A second full factorial held width at 8. Each arm received identical simulator training and evaluation draws within its seed. Each fit used 32 Adam updates with 12 frames per update, the signed P4-A joint loss and its mixed amplifier and SNR schedule. All 48 fits used the development seeds 71301–71303.

Evaluation used 96 held-out 32-subcarrier frames per seed from the equal-power four-tap channel, strong Rapp PA and 10 dB SNR. Each frame was evaluated at gains 1/64, 1 and 64, scaling the noise variance by gain squared. The three gains reuse the same bits. The independent replicate is the seed.

| Enabled feature | BER reduction across three gains, matched parameters | BER reduction at unit gain, matched parameters | BER reduction across three gains, fixed width |
|---|---:|---:|---:|
| Pilot gauge and RMS normalization | 2.442 pp | 0.136 pp | 2.505 pp |
| Hermitian attention with complex rotary encoding | 0.700 pp | 0.234 pp | 0.816 pp |
| Bidirectional pilot/data propagation | −0.119 pp | −0.012 pp | −0.235 pp |

Each effect averages the four matched on/off differences. Positive values mean fewer errors. For the full model, 24,771 of 110,592 gain-conditioned decisions were wrong (22.399%). Disabling bidirectional propagation alone yielded 24,588 errors (22.233%) at the matched budget. The gain-averaged joint effect was negative in two seeds and positive in one. Hermitian attention and rotary encoding change together. The results support conclusions about that combined switch. Three seeds give limited precision.

The three individual seed effects were +2.279, +2.759 and +2.289 percentage points for gauge normalization, +0.606, +0.791 and +0.704 for Hermitian attention, and −0.319, −0.184 and +0.145 for joint propagation. At unit gain, the existing classical receivers achieved lower BER and bitwise binary cross-entropy on the same held-out frames.

| Receiver at unit gain | Bit errors / 36,864 | BER | Mean binary cross-entropy, nats per bit |
|---|---:|---:|---:|
| Full EM-FloatFormer | 8,257 | 22.399% | 0.46812 |
| All three switches off | 8,371 | 22.708% | 0.47237 |
| Bussgang-LMMSE | 6,766 | 18.354% | 0.37090 |
| Two-pass DD-PA | 7,101 | 19.263% | 0.40280 |

## Attention and numerical behavior

For each attention head, mean attention mass was computed by circular frequency lag and correlated with the analytic channel covariance magnitude from the simulator's power delay profile. Self-pairs were excluded. In the first development seed the trained pilot self-attention correlation averaged 0.540 over the two heads on the four-tap channel. It was already 0.544 at initialization. On the sparse long-delay channel these correlations were 0.108 and 0.130, respectively. The attention measurements show appreciable profile dependence and a contribution from initialization. They do not establish that training recovered the physical Green function. Per-layer/head records include key-position permutation controls.

The signed receiver promotes its internal arithmetic to complex128/float64. Its complex64 input and FP32 logit boundary gave a maximum absolute logit change of 1.91 × 10⁻⁶ and zero hard-decision flips across five tested gains spanning 10⁻⁸ to 10⁸. CPU autocast also gave identical hard bits on 100 frames over 0–20 dB. These measurements establish stability on the promoted computation path.

Direct FP16 storage caused 2,004 of 4,096 decisions to change at gain 10⁻⁸. At gain 10⁸, 2,080 input entries overflowed. A separate experiment normalized observations in FP32 first, rounded them to FP16 or BF16 and used the selected precision for all real matrix multiplications inside complex linear projections. Norms, attention, combination of projection terms and demapping retained wide arithmetic. On the 32-frame probe, the FP16 variant changed zero decisions, with maximum logit discrepancy 0.023. BF16 changed two decisions at each scale, with maximum discrepancy 0.186. Repeated gains share the same test frames.

For a single frame, median CPU inference times were 17.31 ms for the trained receiver, 1.94 ms for Bussgang-LMMSE and 7.60 ms for two-pass DD-PA. Measurements used one CPU thread, three warm-up calls, and 30 interleaved timed calls per method. Other research processes shared the host. The output includes batch-eight measurements and timing quantiles.

## Replay

Run from the Antigravity repository root using the same PyTorch CPU runtime. The first two commands generate the 24 matched-budget and 24 fixed-width records. The third regenerates attention, precision and timing measurements.

```powershell
python -m p4.factorial development --threads 1
python -m p4.factorial fixed-width --threads 1
python -m p4.ablation_probing --threads 1
python -m p4.audit_p4c
```

The numerical outputs are `p4/results/p4c_factorial.json`, `p4/results/p4c_fixed_width.json`, `p4/results/p4c_mechanism_probes.json` and `p4/results/p4c_independent_replay.json`. The audit uses a separate initialization/training loop and count calculation. It reproduced the 72 gain-specific integer counts for all 24 matched-budget fits and independently reconstructed both full sets of eight-cell contrasts. Every parameter tensor received a finite gradient at every step and a nonzero gradient at least once per fit. Only development seeds were used. The reserved identifiers in the initial protocol file cannot be consumed by the development entry points.

