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
| P1-E | New-seed replication, exact-sign tests, latency audit; **completed** | [`P1E_REPORT.md`](P1E_REPORT.md), `p1e_results/` |

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

### P1-E: fresh-seed replication (five **new** training seeds)

P1-E trained four model configurations (CNN and Transformer × raw-IQ and log/phase), using seeds **404, 505, 606, 707, 808** and unchanged P1-C training settings. Across five seeds and 2048 frames/seed/test condition (655,360 data bits per receiver/scenario), at 10 dB:

| Receiver | Train3 BER | Unseen equal4 BER | Unseen long6 BER |
|---|---:|---:|---:|
| CNN, log/phase | 5.321% | **6.597%** | 8.564% |
| Transformer, log/phase | 5.251% | 7.410% | 9.557% |
| LS | 8.075% | 8.061% | **8.071%** |
| Matched-profile LMMSE (privileged prior) | **4.777%** | **4.909%** | **5.131%** |

The prespecified CNN-vs-Transformer differences on equal4 and long6 both favored CNN on all five new seeds. **Two-sided exact sign-flip tests with Holm adjustment yielded p=0.125 for each**, so we do **not** claim confirmatory statistical significance. New-seed replication is not independent reimplementation or hardware replication. Complete results, 0/20 dB tests, CPU latency and limitations: [`P1E_REPORT.md`](P1E_REPORT.md).

## Reproduction

Requires Python 3.11+ and PyTorch (CPU is sufficient). The results here were generated with Python 3.13, PyTorch 2.10 CPU, four CPU threads. `matplotlib` is only required to regenerate plots.

```bash
python -m pip install -r requirements.txt
python -m unittest test_p1d_invariants test_p1e_integrity -v
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

# P1-E: new training seeds, then held-out test and exact checkpoint replay
python p1c_architecture.py --steps 750 --batch-size 64 --eval-frames 2000 \
  --seeds 404 505 606 707 808 --arches circular_cnn transformer \
  --encodings raw_iq_magnitude logamp_phase --out p1e_train
python p1e_replication.py --checkpoints p1e_train --out p1e_results --frames 2048
python p1e_replication.py --verify --checkpoints p1e_train --out p1e_results --frames 2048
python p1e_make_report.py
```

The included CSV and figures are **recorded results**, not generated examples. Report scripts read the generated CSV and overwrite reports/images for current runs. Random seeds, scenarios and numerical settings are included as JSON. Full set of 24 P1-C and 20 P1-E weights are distributed separately in the full experiment archive, not in the source repository.

## Experimental assumptions and boundaries

- 32 subcarriers, *all-subcarrier known pilot* for a single symbol, one QPSK data symbol, static channel within each pair; cyclic prefix assumed sufficient for the channel support.
- Ideal frequency-domain model with i.i.d. complex Gaussian thermal noise; known **reported** noise variance. No complete time-domain RF chain and no coding or real antennas.
- Neural receivers share the same hand-crafted pilot-based coarse equalizer and other engineered features. This is an architecture/representation comparison, not an unfettered end-to-end waveform model.
- P1-D tests **only storage quantization of IQ** before FP32 inference, not full FP16/BF16 model inference. The simulated pilot/data phase rotation is **not** full carrier-frequency or sample-clock offset. Sampling-rate robustness is **unverified**.
- A correct-test-prior LMMSE reference is informative but uses an advantage unavailable to the learned model. `true_channel_reference` is an unattainable instantaneous-state reference.
- P1-C/P1-D had three training seeds. P1-E adds five fresh training seeds and prespecified two comparisons with Holm correction; its exact nonparametric p-values are 0.125 (not significant at 0.05). Secondary comparisons remain exploratory; no universal-superiority claim.

## Project files

```text
ofdm_baseline.py               P1-A: generator and classical estimates
p1b_representation.py         P1-B: numerical representation & Transformer
p1c_architecture.py           P1-C: model ablations, training, evaluation
p1c_verify_checkpoints.py     P1-C: reload and repeat saved evaluation
p1c_make_report.py            P1-C: report and figures
p1d_robustness.py             P1-D: frozen-weight robustness evaluation
p1d_make_report.py            P1-D: report and figures
p1e_replication.py            P1-E: new-seed evaluation + exact test + latency
p1e_make_report.py            P1-E: report derived from measured CSV/JSON
test_p1e_integrity.py          P1-E: automated data-consistency checks
P1E_PROTOCOL.md               Prespecified confirmatory comparisons
test_p1d_invariants.py        Unit checks, no checkpoint required
P1B_REPORT.md                 P1-B mathematical + experimental report
P1C_REPORT.md                 P1-C architecture report
P1D_REPORT.md                 P1-D robustness report
P1E_REPORT.md                 P1-E independent new-seed replication report
*/ber_*.csv                   Recorded measurements
*/run_config.json             Reproduction settings
```

**Licensing:** No license is assigned yet. Viewing a GitHub repository does not grant rights to reuse the code; the owner may choose a license separately.

## 中文概述

这是一个**仅使用代码生成的民用 QPSK/OFDM 被动接收机仿真研究**。P1-A至P1-E均已完成。P1-D 在不重新训练的前提下，检验了多个陌生多径分布、接收 IQ 的低精度存储量化、噪声方差报告偏差及相位漂移。P1-E 重新训练了五个全新随机种子，检验模型架构差异的可重复性，并评估统计显著性和CPU推理成本。

**结论具有明确边界：** 在部分未知信道下 CNN 优于小型 Transformer，但六径测试中 CNN 反而劣于普通 LS；知道测试信道先验的 LMMSE 仍较强。P1-E 的两项预设比较虽然在五个新种子中方向一致，但精确检验经 Holm 校正后 p=0.125，未达到0.05门槛。模型不包含真实无线电硬件、发射控制或军事用途的算法。详细数据和局限性见各阶段报告。
