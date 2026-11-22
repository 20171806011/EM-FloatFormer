# 高频与宽带公开实测无线数据集调研与样本验证目录 (`DATASET_DISCOVERY.md`)

> **调研动机**：本次核心实测基线数据集 `rikluost/sdr_ofdm_dataset_v1` 中心频率为 **435 MHz**（室内/室外行人移动，带宽 ~1.5 MHz，128-FFT，CP=6），物理上存在两个结构性局限：(1) 435 MHz 载波波长较长（$\lambda \approx 0.69\text{ m}$），室内行人移动多普勒极小（实测单个 TTI 14 符号内残余相偏标准差仅 `< 0.011 rad`，跨 TTI 存在强准静态硬件/LOS 响应 $|h^{\mathrm{H}}(t)h(t+\Delta)| \approx 0.91$）；(2) 窄带信道 96.2% 能量集中于 2–3 个主径抽头，频率选择性温和。  
> 为支撑 EM-FloatFormer 后续向 **2.4/5 GHz Wi-Fi OFDM、3.75 GHz 5G NR、5.9 GHz V2X、6–22 GHz FR3** 等高频宽带、强频选与高移动性实测场景迁移，本目录系统调研了 Zenodo、Mendeley Data 与 GitHub 上的公开实测射频数据集，并在 Codespace 中拉取了 **3 份开放直链样本（总计 ~49 MB）完成 SHA-256 校验与底层二进制/CSV 解析审计**。

---

## 一、高频/宽带公开实测无线数据集结构化目录

| # | 数据集名称与作者/年份 | 持久标识（DOI / 仓库 URL） | 许可 | 频段与带宽 | 硬件与波形 | 总体积 / 单文件体量 | 是否有标签/导频与可复用性评估 |
| :-: | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **802.11ax Wi-Fi CSI Measurements for Wireless Sensing** (Tonini, Lo Cigno, Gringoli, Cominelli; *IEEE WCNC 2025*) | DOI: [`10.5281/zenodo.17225694`](https://zenodo.org/records/17225694) | CC-BY-SA-4.0 | **5 GHz**；**20 / 40 / 80 MHz** 宽带 OFDM | 商用 Wi-Fi 6 (IEEE 802.11ax) + Nexmon CSI 提取（PCAP / UDP 5500） | `10.04 GB`（模块化按人数分包，最小 `20.87 MB`） | **高复用（信道估计/跨帧预测/频选建模）**：无比特载荷标签，但提供真实 80 MHz 256-FFT 商用 Wi-Fi 6 复数 CSI，含多人室内动态轨迹；**已实测拉取验证**。 |
| **2** | **Intel Wi-Fi CSI Respiratory Sensing Dataset** (Zhu; *EU Horizon MultiX*, 2026-05-31) | DOI: [`10.5281/zenodo.20122379`](https://zenodo.org/records/20122379) | CC-BY-4.0 | **5 / 6 GHz**；**20 MHz (52 sc) & 160 MHz (416 sc)** | Intel Wi-Fi NIC 复数 CSI CSV（含 `OneSittingOneWalking` 行人走动） | `157.47 MB`（单文件 `4.04–59.18 MB`） | **中-高复用（需警惕 160 MHz 子文件缺陷！）**：20 MHz 样本 100% 真实有效（2,130 唯一帧）；**实测审计检出 `160MHz_OneSittingOneWalking.csv` 全部 1,566 行 CSI 重复为同一帧（见第二节审计）**。 |
| **3** | **Datasets of Indoor Wireless Channel Measurements for ML Applications** (Pastore & Ghani, 2021) | DOI: [`10.5281/zenodo.4895133`](https://zenodo.org/records/4895133) | CC-BY-4.0 | **433 MHz / 708 MHz / 2.45 GHz** 三频段同环境实测 | USRP 室内实测点对点原始基带 IQ（QPSK & Gaussian，0/10/20 dB SNR） | `3.63 GB`（20 个独立 `.tar.gz`，每包 `183–205 MB`） | **极高复用（跨频段泛化与物理层解调）**：同一实验室、同一 SDR 平台横跨 433 MHz 至 2.45 GHz，且含已知 QPSK 发射符号与三档 SNR；单包 `< 200 MB` 极易接入。 |
| **4** | **IQ Samples of LTE, 5G NR, Wi-Fi, ITS-G5, and C-V2X PC5** (Merkebu, 2024) | DOI: [`10.5281/zenodo.10811147`](https://zenodo.org/records/10811147) | CC-BY-4.0 | **2.4 / 5.9 GHz**（5G NR / ITS-G5 802.11p / C-V2X / Wi-Fi / LTE，10 MHz） | USRP X310 + OpenAirInterface (5G NR) + openwifi + srsRAN | `1.19 GB`（单包 `dataset_multi_RAT.zip`） | **中高复用（多制式干扰/NBI 与真实 5.9 GHz 波形）**：每个制式 7,500 条真实 IQ 簇，含真实 ITS-G5 / 5G NR / Wi-Fi 频谱共存波形，适合构造真实异构同频干扰源注入 OTA OFDM。 |
| **5** | **n78 Band Live 5G NR Cell IQ Recordings (Vodafone / Orange / Movistar)** (Estévez, Bernal-Escobedo; Universidad de Murcia, 2025) | DOI: [`10.5281/zenodo.15966589`](https://zenodo.org/records/15966589) (Vodafone) / `15966485` / `15966348` | CC-BY-4.0 | **3.75 GHz (n78 商用 5G NR)**，**122.88 Msps** 宽带 | USRP N310 双通道室外天线录波，标准 SigMF 格式 | `28.08 GB`（双通道各 `14.04 GB`） | **高复用（真实商用 5G NR 物理层 DMRS/SSB 接收）**：无发射比特真值，但标准 3GPP 5G NR SSB/PDCCH/PDSCH DMRS 导频结构公开已知，可用于真实商用基站宽带信道估计与 CFO/同步压力测试。 |
| **6** | **FR3 Coherent Multiband Monostatic Channel Measurements** (Pegoraro, Tagliaferri, Mezzavilla et al., 2025) | DOI: [`10.5281/zenodo.17100726`](https://zenodo.org/records/17100726) | CC-BY-4.0 | **6–22 GHz (FR3 厘米波)**，**500 MHz / 1 GHz** OFDM 导频带宽 | 相干多频段实测 CFR（475 组：200 校准 + 150 单目标 + 125 多目标含人体） | `3.17 GB`（含 `data_loader_zenodo.py` 与元数据表） | **高复用（6G 超宽带离格时延与稀疏多径物理建模）**：极宽带 OFDM 导频实测复数 CFR，是检验 EM-FloatFormer 离格连续时延（P12 副轴）与大带宽频率选择性的理想外部物理源。 |
| **7** | **Real-World IQ Dataset for AMR under Multipath Channels** (Mendeley Data) & **NIST ATLAS-Spectrum** (GitHub) | [Mendeley `z3k3s59r7w/2`](https://data.mendeley.com/datasets/z3k3s59r7w/2) · [GitHub `usnistgov/ATLAS-Spectrum`](https://github.com/usnistgov/ATLAS-Spectrum) | CC-BY-4.0 / Public Domain | **2.4 GHz & 5 GHz**（OFDM/QAM/Wi-Fi/LTE） | 真实 LOS 与多径实测基带 IQ（HDF5, 1024-sample 帧）及 NI TDMS/SigMF | 数百 MB–数 GB（按场景分片） | **中等复用**：Mendeley 数据含 2.4 GHz 真实多径 OFDM IQ 帧，适合盲调制/盲 SNR 评估；ATLAS-Spectrum 提供标准化实测射频读取管线。 |
| **8** | **V2X Measurement on M0 Motorway with 4 Vehicles** (BME Automated Drive, 2023) | DOI: [`10.5281/zenodo.8429962`](https://zenodo.org/records/8429962) · [GitHub `BMEAutomatedDrive/v2xdatasets`](https://github.com/BMEAutomatedDrive/v2xdatasets) | CC-BY-4.0 | **5.9 GHz** (IEEE 802.11p / ITS-G5 高速公路实测) | 4 辆实车 Cohda Wireless MK5 OBU（真实高速多普勒 V2V/V2I） | `5.33 GB` (`v2x-dataset.zip`) | **中低物理层复用（高速宏观信道参考）**：以解调后 PHY/MAC 统计与 GPS 轨迹为主（无原始 IQ），适合校准高速 V2X 场景下的真实多普勒与丢包模型参数。 |

---

## 二、Codespace 实测样本拉取、SHA-256 校验与底层解析审计

遵循 `< 200 MB` 轻量样本验证规范，我们在 Codespace `/tmp/agy_cli3_ofdm_stress/raw/extra_datasets/` 实际下载并解析了 **2 个高频宽带数据集的 3 个原始文件（合计 48.4 MB，不提交原始大文件至 Git 仓库）**：

### 2.1 文件完整性清单（SHA-256 实测）

| 样本文件 | 来源 DOI | 下载体积 | 实测 SHA-256 | 解析状态与数据卫生裁决 |
| :--- | :--- | ---: | :--- | :--- |
| `wifi6_80MHz_5people.zip` | `10.5281/zenodo.17225694` | `20.87 MB` (`21,881,182` B) | `d32eb9a248aeb94df595d3382a5720a8dd108604d224e247f9a4e66862df258a` | **PASS（高质量可用）**：成功解包 `5people/2024-08-01/config.json` 与 `capture0.pcap`（`11,893,836` B）；含 `10,440` 个标准 Nexmon UDP 5500 帧（源 MAC ASCII `NEXMON`），**`10,440 / 10,440`（100%）CSI 载荷字节完全唯一**，长度 `1110 / 1162` B（末尾 `1024` B 为 80 MHz 802.11ax 256-FFT 复数 CSI）。 |
| `CSI_20MHz_OneSittingOneWalking.csv` | `10.5281/zenodo.20122379` | `4.04 MB` (`4,234,033` B) | `39e1b03321206f61aeef43343c78979fc7e714466b6ea802b750137d23ea7a8c` | **PASS（高质量可用）**：成功解析 `2,130` 帧 × `52` 个复数子载波，**`2,130 / 2,130`（100%）帧唯一**；总录制时长 `39.21 s`，中位帧间隔 `16.0 ms`（~62.5 Hz 采样率）。 |
| `CSI_160MHz_OneSittingOneWalking.csv` | `10.5281/zenodo.20122379` | `23.51 MB` (`24,648,384` B) | `ceb6861339c0b21844292e9cd191428c0d728a63b6ef309ef462b0badd3a94a6` | **WARN（检出源数据导出缺陷！）**：共 `1,566` 行 × `416` 个复数子载波（时间戳递增 28.22 s），但 **唯一 CSI 向量数 = `1`**（全部 1,566 行的 832 列实/虚部数值逐位完全相同，行范数恒为 `93098.68308`！）。单帧频域响应本身真实呈现 160 MHz 强频选性（`freq_amp_cv = 0.2130`，相隔 16 子载波相关系数降至 `0.7415`），但时间维已被上游导出脚本冻结，**严禁直接用于时序预测/跨帧学习**。 |

### 2.2 435 MHz 窄带基准 vs 5 GHz 宽带实测信道物理特性横向对比

| 物理维度 | `sdr_ofdm_dataset_v1` (435 MHz, 主研究数据集) | `Zenodo 20122379` (5/6 GHz, 20 MHz 实测验证) | `Zenodo 20122379` (160 MHz 单帧频域特征) | `Zenodo 17225694` (5 GHz, 80 MHz 802.11ax 验证) |
| :--- | :--- | :--- | :--- | :--- |
| 有效子载波数 / 带宽 | 101 非 DC 子载波（~1.5 MHz 窄带） | 52 子载波（20 MHz） | 416 子载波（160 MHz） | 256-FFT 80 MHz 802.11ax（1024B/帧） |
| 频域幅值变异系数 `freq_amp_cv` | 低-中等（CP=6 内含 96.2% 能量） | `0.0545`（20 MHz 室内平坦） | **`0.2130`**（近 4 倍强频率选择性） | 80 MHz 宽带强频选（商业 AP 实测） |
| 频域子载波相关 `df=16` | 高相干（长相干带宽） | `0.9932` | **`0.7415`**（明显多径频选衰落） | 高频宽带多径结构 |
| 去均值幅值时域自相关衰减 | `lag=1: +0.095` → `lag=20: -0.020`（20 TTI 退相干） | `lag=1 (16ms): +0.382` → `lag=50 (0.8s): +0.287` → `lag=200 (3.2s): +0.241` | *不适用（上游导出冻结，仅 1 唯一帧）* | `10,440` 唯一动态帧（`usleep=22 ms`，5 人室内） |
| 是否有发射比特真值 | **是**（1,400 RE × 4 bits = 5,600 bits/TTI） | 否（仅 CSI + SNR） | 否（仅 CSI + SNR） | 否（仅 Nexmon 802.11ax CSI） |

---

## 三、对 EM-FloatFormer 下一阶段实测路线的建议

1. **近期最高性价比实测扩展（端到端 BER/SER 物理层跨频段验证）**：优先接入 **`Zenodo 4895133`** 的 `f2450_snr10_QPSK.tar.gz`（188.67 MB）与 `f433_snr10_QPSK.tar.gz`，在同一硬件、同一实验室下直接检验 EM-FloatFormer 从 433 MHz 到 2.45 GHz 的跨频段鲁棒性。
2. **近期宽带频率选择性与离格时延验证**：利用已验证 100% 唯一的 **`Zenodo 17225694`**（5 GHz 80 MHz 802.11ax，10,440 帧）和 **`Zenodo 17100726`**（6–22 GHz FR3 500 MHz/1 GHz CFR）作为真实宽带物理信道冲激响应库，叠加真实 16-QAM/64-QAM OFDM 波形与受控多普勒，验证离格延迟估计与宽带稀疏导频内容寻址。
3. **数据卫生铁律**：本次对 `Zenodo 20122379` 160 MHz CSV 检出“1,566 行仅 1 个唯一 CSI 向量”再次证明——接入任何外部公开实测数据集前，必须执行 **SHA-256 校验 + 唯一帧数检查 + 星座/物理守恒自检**，绝不可盲信元数据标题直接训练。
