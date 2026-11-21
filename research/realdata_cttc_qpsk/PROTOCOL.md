# EM-FloatFormer 实测专项实验协议（CTTC 室内多频段多信噪比 QPSK）

- **项目基线提交**：`f566f9b8b0ddfb468a4cb3fd64588bfa46c87212`（分支 `research/p12-first-principles-diagnostics`）
- **本专项工作分支**：`research/realdata-cttc-indoor-qpsk`
- **本专项交付目录**：`research/realdata_cttc_qpsk/`
- **理论与源码衔接**：
  - `p12/P12_CONSISTENCY_FIX.md` 与 `p12/offgrid_receiver.py`（单次计分先验一致性、防泄漏留出验证、分折/分组隔离）
  - `research/content_addressing_scalar/REPORT.md`（固定局部窗口下界、全局内容寻址精确风险公式 (2)、范围增益 vs 内容识别方差分解公式 (7)、过冷/过热与共享观测偏差反例）
  - `research/posterior_fusion/pages/01.md–12.md`（信道后验均值正交投影定理、总协方差分支内+分支间分解、MSE 与 BER 排序反转反例、完整混合后验进入逐符号/逐比特 MAP 与 log-sum-exp LLR、平均 LLR 负对照）

---

## 1. 数据来源与完整性验证

- **官方数据集**：Adriano Pastore, Armin Ghani (Centre Tecnològic de Telecomunicacions de Catalunya, CTTC), *Datasets of Indoor Wireless Channel Measurements for Machine Learning Applications* (v1.0, ARISTIDES Project Grant RTI2018-099722-B-I00), Zenodo, June 2, 2021.
- **DOI**：`10.5281/zenodo.4895133`
- **官方链接**：`https://zenodo.org/records/4895133`
- **API 端点**：`https://zenodo.org/api/records/4895133`
- **许可协议**：CC-BY-4.0
- **纳入实测会话**：全量 **3 频段 × 3 名义 SNR = 9 个 QPSK 实测采集会话**（全部于 2021-05-19 在 CTTC 室内静态实验室由 Ettus Research USRP X310 + VERT2450 单垂直全向天线录制），以及 `README.pdf` 与 `sources.tar.gz`（含 `source_QPSK.dat`）。全部原始压缩包与解压文件的精确大小、官方 MD5 与实测 SHA-256 记录于 `data_manifest.json` 及 `download_cttc.py`。

### 1.1 波形与物理处理阶段审计
1. **发送端**：`source_QPSK.dat` 为 1,000,000 个随机字节（`uint8`，取值 `{0, 1}`），按 Gray 码映射为 500,000 个单位能量复数 QPSK 符号 $x_k = \frac{1}{\sqrt{2}}[(1-2b_{2k}) + j(1-2b_{2k+1})]$，循环重复发射（`Repeat: Yes`，周期恰为 1.0 s）。符号率 $f_{\text{symb}} = 0.5\text{ MS/s}$（1 Mbps，无信道纠错编码，故**原始编码位 BER = 信息位 BER**），TX RRC 脉冲成形滚降系数 $\alpha = 0.2$、256 taps、2× 插值（$f_{\text{sample,TX}} = 1\text{ MS/s}$）。
2. **接收端**：RX 采样率 $f_{\text{sample,RX}} = 2\text{ MS/s}$（标称过采样因子 $\text{SPS} = 4$ samples/symbol）。每个会话录制约 11.001 秒（22,001,908 个 `complex64` 样本），开头约 1 秒发射机静默用于背景噪声方差估计。
   - `file_1.dat`：ADC 下变频后原始复基带 I/Q（未过 RX RRC 匹配滤波）。
   - `file_2.dat`：经 GNU Radio 256-tap RRC 匹配滤波（Decimation = 1，仍保持 4 samples/symbol）后的复基带 I/Q。
3. **一手实测物理勘误与特性发现**：
   - **比特奇偶对齐偏移**：7/9 会话为偶数比特对齐（`bits[0::2], bits[1::2]`），但 `f433_snr20_QPSK` 与 `f433_snr0_QPSK` 存在 1 比特奇数偏移（`bits[1::2], bits[2::2]`）。`adapter.py` 通过自动互相关锁定唯一对齐方式并记录 `bit_alignment`，使全部 3 个 `snr20` 会话均达到零误码基线。
   - **晶振采样率偏差（SFO）与载波频偏（CFO）**：全会话实测存在 `+0.39 ~ +0.55 ppm` 的真实晶振时钟偏差（10 秒漂移 > 2 个符号）及对应载频比例 CFO（2450 MHz 约 `-1175 ~ -1285 Hz`，708 MHz 约 `-370 ~ -395 Hz`，433 MHz 约 `-157 ~ -218 Hz`）。
   - **真实硬件杂散/多普勒双峰**：`f708_snr10_QPSK` 实测存在幅度高达主峰 `26.0%`、频偏 `+1510.6 Hz` 的相干边带，引发 `16.5%` 幅度拍频与剧烈瞬时频偏摆动。

---

## 2. 标准化数据包切分与四重零泄漏防火墙

为消除 `source_QPSK.dat` 在 ~9 秒录制内循环重复及跨会话复用导致的潜在序列/窗口泄漏，强制执行以下不可违背规则：

1. **包结构（$L_{\text{pkt}} = 256$ 符号 = 512 比特 = 0.512 ms）**：
   - **同步头（`SYNC_IDX`）**：索引 `0..5`（6 符号，接收机可见）
   - **稀疏包内导频（`PILOT_IDX`）**：索引 `126, 127, 254, 255`（4 符号，接收机可见；与同步头合计 `KNOWN_IDX` = **10 符号**，导频开销 `3.91%`）
   - **留出验证导频（`HOLDOUT_IDX`）**：索引 `40..47` 与 `200..207`（16 符号，**严格留出，接收机不可见**，专用于独立评估留出信道估计 NMSE 与高斯预测 NLL）
   - **载荷（`PAYLOAD_IDX`）**：其余 **230 符号**（460 原始/信息比特，**接收机完全盲测**）
   - 每个包前后附带 `PAD_SYMS = 8` 符号（32 RX 样点）保护波形用于分数定时插值与多抽头均衡。
2. **源比特区间绝对互斥（含 4,096 符号保护带，覆盖 EVEN/ODD 对齐偏移）**：
   - `train` 目标包：`[0, 160_000)`；`train` 专属参考池：`[164_096, 200_000)`
   - `dev` 目标包：`[204_096, 290_000)`；`dev` 专属参考池：`[294_096, 330_000)`
   - `test` 目标包：`[334_096, 455_000)`；`test` 专属参考池：`[459_096, 495_904)`
   - 环绕保护带：`[495_904, 500_000)`（4,096 符号）。
3. **时间间隔隔离**：每个会话在稳定区间（$t \ge 1.45\text{s}$）抽取连续 8 个 1 秒循环（`loop_id = 0..7`，9 会话 × 8 循环 = **72 个独立会话-循环簇 `cluster_id`**）；同循环内相邻入选包步长 $\ge 2,048$ 符号（包间空隙 $\ge 1,792$ 符号 $> 28\times$ RRC 滤波器长度），彻底杜绝滤波拖尾与短时衰落重叠。
4. **固定测试清单规模**：
   - 原始实测：`train = 1,440` 目标包（+ 576 参考包），`dev = 864` 目标包（+ 576 参考包），`test = 1,440` 目标包（+ 576 参考包，总计 331,200 载荷符号 = 662,400 比特）。
   - 独立可控物理扰动压力测试（在原始实测 IQ 上注入 3 类扰动：`stress_low_snr`、`stress_cfo_phase_jump`、`stress_burst_impairment`）：`train = 720`、`dev = 432`、`test = 1,080` 包（每类扰动 360 包，覆盖全部 9 会话与 72 簇），结果与原始实测完全分轨保存。

---

## 3. 接收机信息可见性与四组配对对照设计

### 3.1 严格信息可见性约束
| 对象 | 接收机可见 | 仅离线评测器可见 |
| :--- | :--- | :--- |
| 目标包 | `y_iq_4x`, `x_known` (10 符号), `KNOWN_IDX`, `bg_noise_var` | `x_all_true`, `HOLDOUT_IDX` 真值, `PAYLOAD_IDX` 真值, `h_offline_ref`, `offline_cfo_hz`, `capture_id` |
| 参考池包 | 仅由参考包自身 10 个已知导频提取的 `RefEntry`（CFO/定时/功率/噪声/SNR/斜率） | 参考包载荷、留出导频与 `capture_id`（仅用于离线审计匹配权重质量） |
| 调优与选模 | 所有经典超参、温度校准、网络训练早停与种子选择严格仅用 `train` / `dev` | `test` 仅在全部冻结后评测一次 |

### 3.2 四组实验口径
1. **成熟经典接收基线**：匹配滤波（`file_2.dat`，并在 `f2450_snr20` 上与 `file_1.dat` 自建 256-tap RRC 交叉复核）+ 分数定时同步 + 导频联合 ML CFO 估计 + `classical_ls` / `classical_mmse` / `classical_mmse_slope` / `classical_mmse_3tap` + QPSK 硬/软解调。
2. **对照 A（全局参考与内容匹配）**：在完全相同的跨会话全局混合参考池（$M=36$，9 会话各 4 包随机打乱，仅前 $k=4$ 为固定局部窗口）及同会话时间参考池（$M=24$，按时间距离排序，近邻 $k=4$ 为局部窗口）上，通过相同下游先验条件接收机配对比较：`classical_mmse`（无外部参考）、`A1_local_window_k4`、`A2_global_uniform`、`A3_global_snr_weight`、`A4_global_content_attn`（精确相位边缘化复高斯导频对数似然 + dev 校准温度 $\tau_A$）、过冷硬检索 `A5_content_cold_hard` 与过热 `A6_content_hot`；同步检验 `content_addressing_scalar` 风险分解与权重距离关系。
3. **对照 B（硬选 vs 线性融合 vs 保留混合后验似然）**：共享完全相同的 $L=12$ 物理竞争候选库（5 个局部多峰 CFO + 3 个互斥全局内容参考簇 + 2 个线性斜率漂移 + 2 个分段中包跳变分支）与完全相同的单次计分边缘对数证据后验权重 $\alpha_r$，配对比较：`B1_hard_select`、`B2_linear_fusion_plugin`（含总方差 = 分支内 + 分支间）、`B2b_linear_within_only`（故意省去分支间方差，检验 §4.1 失准预言）、`B3a_sym_mixture`（逐符号混合）、`B3_mixture_likelihood`（精确包级 LOO 联合混合后验进入 QPSK MAP 与 log-sum-exp LLR，验证 §5.2）与 `B4_avg_llr_antipattern`（负对照，验证 §5.2 平均 LLR 缺陷）。
4. **对照 C（轻量级可训练 Attention / 权重预测 / 残差网络）**：固定上述全部 $L=12$ 物理候选生成与后端 `mixture_likelihood` 判决器，在单线程 PyTorch CPU（`TORCH_NUM_THREADS=1`）下使用 3 个独立随机种子 `{20261121, 20261122, 20261123}` 训练 `C1_attn_weight_net`（跨候选自注意力残差权重网络）与 `C2_attn_weight_resid_net`（权重网络 + 有界复增益/相位/对数方差残差头），以可微混合 Bit BCE + 训练留出信道 MSE 联合优化，在 `dev` 上早停选模，在 `test` 上与 dev 调优后的最强经典方法 `B3_mixture_likelihood` 严格配对比较。

---

## 4. 统一指标定义与统计推断

- **分段统计**：分别输出 `sync`（6 符号）、`pilot`（4 符号）、`holdout_pilot`（16 符号）、`payload`（230 符号 = 460 比特）的指标。
- **核心指标**：SER、符号重建 MSE（raw）、原始 EVM 与无偏 EVM（消除线性收缩偏置）、BER（明确标注无信道编码：原始位 = 信息位）、**PER / FER**（事件定义：单包 230 个载荷符号 / 460 比特中存在 $\ge 1$ 个判决错误即计为 1 次包错；分母为该集总有效评估包数）、留出导频观测预测 NMSE（`holdout_obs_nmse`）、相对离线高精度参考信道的留出 NMSE（`holdout_ref_nmse`）、留出高斯信道预测 NLL（`holdout_ch_nll_nats`）、载荷软比特 NLL（`payload_bit_nll_nats`）与 15-bin ECE（`payload_bit_ece`）。
- **配对统计推断**：以 72 个独立会话-循环簇（`cluster_id`）及 9 个采集会话（`capture_id`）为独立抽样单元，计算簇级配对改善差值（正值表示候选方法优于基线）、胜/平/负计数、10,000 次分层簇 Bootstrap 95% 置信区间，以及精确（$G \le 16$ 时 $2^G$ 态）或 50,000 次蒙特卡洛配对符号翻转 $p$ 值。
- **工程成本与鲁棒性**：统一单线程（`OMP=MKL=OPENBLAS=TORCH=1`）记录每包中位/均值/P95 推理耗时、训练时间、参数量、进程峰值 RSS 内存、成功/失败/异常数及处理覆盖率。
