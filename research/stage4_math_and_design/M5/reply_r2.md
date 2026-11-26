# M5 Round 2 · Adaptive reference selection and the Stage 6 lock

The 16-bit cache supports recording-specific reference selection across **jump interval, selective fade, and CFO alias** geometries. All four candidate IDs win on held-out recordings within every geometry. Old IQ identifies where to look while leaving the two hypotheses equally likely.

## CPU results

The pinned uv replay completed on **12,288 blind mothers** after a separate 6,144-mother dev split. Fixed-best-on-dev minimizes observed post-reveal NLL within each public family, freezing candidate IDs **3, 1, 2** for jump, fade and CFO respectively. The oracle can choose the best constant retrospectively per family and endpoint. Each policy uses one reveal.

| Policy | Post-reveal NLL (nat) | Accuracy | Max-EIG NLL gain and simultaneous 95% CI | Max-EIG accuracy gain and simultaneous 95% CI |
|---|---:|---:|---:|---:|
| **Max EIG** | **0.04823** | **98.20%** | — | — |
| Random | 0.51722 | 64.96% | 0.46900 [0.45756, 0.48043] | 33.24 [31.87, 34.61] pp |
| Nearest | 0.51098 | 66.06% | 0.46275 [0.45132, 0.47419] | 32.15 [30.78, 33.51] pp |
| Fixed best on dev | 0.51010 | 66.23% | 0.46187 [0.45034, 0.47340] | 31.97 [30.62, 33.33] pp |
| Fixed oracle | 0.50672 | 66.38% | 0.45849 [0.43888, 0.47799] | 31.82 [29.47, 34.17] pp |

All 30 elementary contrasts were covered by simultaneous paired intervals. Each family clears both endpoints against each of its four constants. Across families, the committed forecasts predict 0.64184 nat of EIG and the selected observations yield 0.64492 nat of realized log-score gain. These effects describe a deliberately structured synthetic benchmark with a calibrated physical reader.

## Exact result

The gain over the population-best public-stratum constant is $E[\max_a U_a-\max_a E(U_a\mid M)]$, where $U_a=I(H;Y_a\mid Z,M)$. It is positive iff every fixed option is strictly inferior with positive probability on a positive-measure set of strata. With unique maximizers this is exactly $I(a^*;U\mid M)>0$. With ties, minimize that information over all maximizing choices. NLL and accuracy have different objectives. This run tests both, and the theorem gives a counterexample to general accuracy dominance.

## Integration lock

- **M4 to M1.** The v1 Stage 6 word is little-endian and preserves the exact s17 high nibble. It uses radial code 0 for erasure. The compatibility adapter merges two low M4 radial cells and keeps all higher edges. It matched M4's actual R3 packer on 188 fixture words and preserved all 65,536 possible prefixes. Measure residual quality before adding R3 ancillary noise.
- **M3.** The selected immutable acquisition ID, timestamp, validated association, calibrated variance and matched sum pass through the same predictive likelihood. Twenty-four held-out reveals replayed through M3's shared pool agreed within $4.0\times10^{-15}$. The raw correlated-Gaussian innovation check agreed within $6.7\times10^{-16}$. Prior IDs and duplicate presentations add zero evidence.
- **M2 and M6.** Use the identical five 16-bit words, public catalogue, one sealed reveal and typed forecast/posterior schema across Qwen3.5-4B, Qwen3.5-9B, Qwen3.8-27B and a qualifying LEMMO adapter. Freeze baseline choices and calibration on dev and use paired NLL and accuracy with host-level multiplicity control. Keep a separate matched-action test for cache precision. The host contract is specified and the CPU physical path is verified. The 4B/9B/27B model runs, matched LEMMO checkpoint/adapter and real SDR qualification are pending.

[Theorems and contract](THEOREM_M5.md#7-round-2--exact-value-of-adapting-to-the-recording) · [Pinned CPU replay](verify_m5.py) · [Results JSON](results_m5.json)
