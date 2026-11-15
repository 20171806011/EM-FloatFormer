# EM-FloatFormer — Synthetic OFDM representation and receiver benchmarks

[中文概述](#中文概述) · [Reproduction](#reproduction) · [Results](#measured-results)

> **Research prototype, not a validated RF foundation model.** This repository contains a receive-only, purely synthetic, civilian QPSK/OFDM benchmark. It contains **no radar, electronic-warfare logic, jammer, transmitting hardware control, weapon targeting, or real traffic data**.

The central experimental question is whether learned representations of complex baseband samples and particular neural architectures improve decoding under ordinary multipath fading and thermal noise, compared with classical signal processing at known computational cost. We publish negative results alongside positive results.

## Status (2026-11-15)

| Milestone | Description | Evidence |
|---|---|---|
| P1-A | Small OFDM simulator + first Transformer and LS/DFT/LMMSE references | `ofdm_baseline.py`, `results/` |
| P1-B | Raw IQ + magnitude vs log amplitude + regularized phase; mathematical numerical checks | [`P1B_REPORT.md`](P1B_REPORT.md), `p1b_results/` |
| P1-C | Token-MLP, MLP-Mixer, 1-D CNN, Transformer with matched parameter budgets, three seeds | [`P1C_REPORT.md`](P1C_REPORT.md), `p1c_results/` |
| P1-D | Frozen-checkpoint robustness: five delay profiles, IQ quantization, noise-report error, common-phase drift | [`P1D_REPORT.md`](P1D_REPORT.md), `p1d_results/` |
| P1-E | Independent expansion / additional trials | **Not started** |

**These results do not demonstrate a universally superior Transformer, novel physical law, real receiver deployment, or full radio foundation model.**

## Measured results

At 10 dB for previously unseen *four equal-power multipath taps* (P1-D, three model seeds, 196,608 evaluated bits for each receiver):

| Model | Bit error rate |
|---|---:|
| Circular CNN, log/phase representation | **6.575%** |
| Transformer, log/phase representation | 7.434% |
| Pilot-only LS | 8.036% |
| LMMSE with *correct test-profile prior* (**additional information**) | **4.842%** |

When the profile is changed to **six taps**, CNN/log-phase BER rises to **8.565%** whereas LS is **8.222%**. The CNN does **not** retain its apparent advantage over the generic classical estimator. More results, 0/10/20 dB curves, and limitations: [`P1D_REPORT.md`](P1D_REPORT.md).

![10 dB results across five channel profiles](p1d_results/profile_robustness_10dB.png)

## Reproduction

Requires Python 3.11+ and PyTorch (CPU is sufficient). The results here were generated with Python 3.13, PyTorch 2.10 CPU, four CPU threads. `matplotlib` is only required to regenerate plots.

```bash
python -m pip install -r requirements.txt
python -m unittest test_p1d_invariants -v
python ofdm_baseline.py --smoke-only
python p1c_architecture.py --smoke
```

Train and assess from scratch (P1-C checkpoints are **not** committed to GitHub; rebuild them):

```bash
python p1b_representation.py --steps 750 --eval-frames 2000 --seeds 101 202 303 --out p1b_results
python p1c_architecture.py --steps 750 --eval-frames 2000 --seeds 101 202 303 --out p1c_results
python p1c_verify_checkpoints.py --dir p1c_results --seed 101 --eval-frames 2000
python p1c_make_report.py
python p1d_robustness.py --smoke
python p1d_robustness.py --frames 1024 --seeds 101 202 303 --out p1d_results
python p1d_robustness.py --verify --frames 1024 --seeds 101 --out p1d_results
python p1d_make_report.py
```

The included CSV and figures are **recorded results**, not generated examples. Report scripts read the generated CSV and overwrite reports/images for current runs. Random seeds, scenarios and numerical settings are included as JSON. Full set of 24 P1-C weights is distributed separately in the local experiment archive when available, not in the source repository.

## Experimental assumptions and boundaries

- 32 subcarriers, *all-subcarrier known pilot* for a single symbol, one QPSK data symbol, static channel within each pair; cyclic prefix assumed sufficient for the channel support.
- Ideal frequency-domain model with i.i.d. complex Gaussian thermal noise; known **reported** noise variance. No complete time-domain RF chain and no coding or real antennas.
- Neural receivers share the same hand-crafted pilot-based coarse equalizer and other engineered features. This is an architecture/representation comparison, not an unfettered end-to-end waveform model.
- P1-D tests **only storage quantization of IQ** before FP32 inference, not full FP16/BF16 model inference. The simulated pilot/data phase rotation is **not** full carrier-frequency or sample-clock offset. Sampling-rate robustness is **unverified**.
- A correct-test-prior LMMSE reference is informative but uses an advantage unavailable to the learned model. `true_channel_reference` is an unattainable instantaneous-state reference.
- Three training seeds and multiple scenario comparisons are exploratory, not a confirmatory claim of universal statistical superiority. No multiple-comparison adjustment.

## Project files

```text
ofdm_baseline.py               P1-A: generator and classical estimates
p1b_representation.py         P1-B: numerical representation & Transformer
p1c_architecture.py           P1-C: model ablations, training, evaluation
p1c_verify_checkpoints.py     P1-C: reload and repeat saved evaluation
p1c_make_report.py            P1-C: report and figures
p1d_robustness.py             P1-D: frozen-weight robustness evaluation
p1d_make_report.py            P1-D: report and figures
test_p1d_invariants.py        Unit checks, no checkpoint required
P1B_REPORT.md                 P1-B mathematical + experimental report
P1C_REPORT.md                 P1-C architecture report
P1D_REPORT.md                 P1-D robustness report
*/ber_*.csv                   Recorded measurements
*/run_config.json             Reproduction settings
```

**Licensing:** No license is assigned yet. Viewing a GitHub repository does not grant rights to reuse the code; the owner may choose a license separately.

## 中文概述

这是一个**仅使用代码生成的民用 QPSK/OFDM 被动接收机仿真研究**。P1-A至P1-D已完成。P1-D 在不重新训练的前提下，检验了多个陌生多径分布、接收 IQ 的低精度存储量化、噪声方差报告偏差及相位漂移。

**结论具有明确边界：** 在部分未知信道下 CNN 优于小型 Transformer，但六径测试中 CNN 反而劣于普通 LS；知道测试信道先验的 LMMSE 仍较强。模型不包含真实无线电硬件、发射控制或军事用途的算法。详细数据和局限性见各阶段报告。
