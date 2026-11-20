# EM-FloatFormer P9 整合研究计划书（融合 P8-C 与 `p8_manifold` 实证）

> **编制依据**：整合 [EM_FloatFormer_P9_Antigravity.md](file:///C:/Users/%E6%9D%8E%E5%88%99%E5%BE%90/Downloads/EM_FloatFormer_P9_Antigravity.md)、远端基线 `P8-C` (`ce2ff50e81f6b529ec0311393da0013b0a85b801`) 与刚刚完成的 `p8_manifold` (`e76c61eef26151985963460b0ab03684cd562e3e`) 全部正式证据。

---

## 0. 双基线登记与已完成的前置核验

| 研究线 | 分支与提交哈希 | 关键已证实结果 | 留给 P9 的核心问题 / 可复用构件 |
| :--- | :--- | :--- | :--- |
| **`P8-C` 跨波形经典基线** | `origin/research/p8c-ofdm-single-carrier` (`ce2ff50e81f6b529ec0311393da0013b0a85b801`) | 9 dB OFDM → CP-SC 稀疏 1 训练块：继承同链路 BLER `54.84%` vs 清空 `89.47%` vs 错配 `98.71%`（Holm `p=0.0078125`）；净正确交付 `+175` 块（`+7.06%`）。真信道参考 BLER 仍高达 `38.93%`（`1,993/5,120`）；稀疏共有 BLER 仍比密集清空高 `+3.24 pp`（超 `+1.00 pp` 非劣界限）。 | 1. 需拆解 `38.93%` 真信道错误的来源（噪声 vs 线性均衡 vs LDPC）并给出严格 ML 下界证书；<br>2. 需让接收机自主辨认历史何时失效（连续 / 部分变化 / 完全变化）；<br>3. 需逐层消除 `p8/link.py` 硬编码 `N0`。 |
| **`p8_manifold` 无字典几何流形线** | `research/p8-coherent-transport-manifold` (`e76c61eef26151985963460b0ab03684cd562e3e`) | `N=10` 冻结正式实验（`5,760` 块/条件，`40/40` 重放通过，`3,240` 次 $\mathbb{C}^*$ 变换 `0` 翻转）：<br>• `O1` BCE 上 `EM8-Transport` 首次显著击败等参 `EM8-Local`（`9/10`, `p_holm=0.02344`），但硬 BLER 持平（`81.67%` vs `81.61%`）；<br>• `O0` 出现双分离：`Transport` 赢 BCE（`p_holm=0.0156`）而 `AbsNoTransport` 赢 BLER（`p_holm=0.0176`），揭示共享单标量 `omega_h` 在多径异号多普勒叠加时的过旋风险；<br>• `O2` 盲 Student-$t$ 重降门实现 **143.8×** 拒绝比（`10/10`, `p_holm=0.00586`）；<br>• `O3` 匹配温度 Schur 补同化全部 6 组正向改善（BCE 与 `4.92x` 选择性均为 `10/10`, `p_holm=0.00781`）；<br>• Scaling Ladder 证实固定训练预算存在 Chinchilla 最优容量（`2K > 13K > 40K`）。 | 1. **直接升级 P9-C**：弃用 `p7/routing_operator.py` 的无符号复注意力，引入 `p8_manifold` 的**有符号 $U(1)$ 平行输运 + 多尺度测地距离先验 + 精确 $\mathbb{C}^*$ 不变 Jeffreys 散度 `D_J(p_inherit || p_clear)`**，同时保留等参 `AbsNoTransport` 与 `Local` 三向对照，且采用 Chinchilla 紧凑容量（`~1K–3K` 实参数、`4` 头避免双头过旋）；<br>2. **直接增强 P9-B**：用预更新 Jeffreys 散度与 Student-$t$ 连续门控补强“部分路径变化（机制 ②）”处理。 |

- **P9 独立研究分支与工作树**：从 `ce2ff50` 创建 `research/p9-cross-waveform-trust-gating`（工作树 `C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-p9`，与 `p8_manifold` 完全隔离，不触碰 `main`）。
- **基线核实已完成（P9 验收项 #1）**：在独立工作树中运行 `p8/tests`（`9/9 OK`）、`p8c/tests`（`7/7 OK`）并重放 `P8-C` 第 1 与第 10 组全部 64 条链路（512 单元），跨平台整数字段差异严格为 **`0`**，最大浮点差异 `8.527e-11`（纯 Windows/Linux OpenBLAS ULP）。

---

## 1. 四路完全独立并行架构（调用原版 `agy-bin.exe` + `gemini-4-neon-xhigh` + `--effort xhigh`）

所有并行 CLI 代理共用只读接口契约 `p9/contract.py`，但各自独占严格不重叠的文件集合，零数据依赖、零写冲突：

| 并行 CLI 会话 | 对应原计划角色 | 独占文件（绝不越界） | 核心科学交付（不依赖其他会话数据即可直接跑通） |
| :--- | :--- | :--- | :--- |
| **CLI-1 (`P9-A`)** | 信道波形组 + 通信算法组 | `p9/channel.py`<br>`p9/decoder.py`<br>`p9/p9a_study.py`<br>`p9/tests/test_p9a.py` | 1. 显式 `n0` 多 SNR（`6/9/12/15 dB`）OFDM→SC 生成器（含 AWGN、零噪声、真三径及 P9-B 三种历史机制 ①连续 ②单径边界相位连续多普勒跳变 ③独立新信道 + OOD 跳变幅度/时刻）；<br>2. 强经典单载波解码器（**Turbo MMSE-PIC + Numba BP + OSD-2 MRB 候选列表 + 精确 32×32 时变全波形残差重排 `‖y-Hx(c)‖²`**），并在开发集对比小/大候选表精度与耗时；<br>3. **数学严格 ML 下界证书**：在评估器内检查 64 个单信息位翻转邻码字 + 实际解码器发现的全部合法码字（含浮点平局保护），并在微型 `(16, 8)` 码本上完全穷举验证证书方向与全波形得分；<br>4. 跑完 P9-A 多 SNR 交叉归因开发实验（真信道/估计信道 × 原生 LMMSE+LDPC / 强解码器，逐块净救回/净损失 + 证书下界 + 每千块 CPU 耗时），落盘 `p9/results/p9a_dev_attribution.json`。 |
| **CLI-2 (`P9-B`)** | 统计推断组 | `p9/gating.py`<br>`p9/p9b_dev.py`<br>`p9/tests/test_p9b.py` | 1. 在 OFDM→SC 两条物理候选（继承推进 A vs 清空重估 B）上实现严格**预更新**经典门控：① 始终继承、② 始终清空、③ **预更新边际预测对数贝叶斯因子门控**（在 B 训练块拟合前计算精确预测似然，绝不用拟合后残差冒充）、④ 分段交叉预测门控、⑤ **融合 `p8_manifold` 构造 B 的 $\mathbb{C}^*$ 不变 Jeffreys 散度 + Student-$t$ 连续门控**（应对部分路径变化机制 ②）；<br>2. 概率空间分支混合（含分支均值间方差 `within + between`）与统一的分支合并整字打分接口；<br>3. 跑完 P9-B 三机制等权（①连续 ②部分变化 ③完全变化）+ OOD + 旧式错配检查开发实验，单报各机制 BLER、帧末 NMSE、校准比与保护连续链路代价，落盘 `p9/results/p9b_dev_gating.json`。 |
| **CLI-3 (`P9-C`)** | 学习架构组 | `p9/neural_gate.py`<br>`p9/p9c_dev.py`<br>`p9/tests/test_p9c.py` | 1. 基于 `p9/contract.py` 公开接口实现严格同参（`~1.5K–2.5K` Chinchilla 紧凑实参数、`4` 头防过旋）且全 $\mathbb{C}^*$ 不变的可学习时间门控：<br>• `P9C-TransportAttn`（有符号 $U(1)$ 平行输运 + 多尺度测地距离 + Jeffreys 偏置）<br>• `P9C-AbsNoTransport`（精确同参无符号/无输运消融）<br>• `P9C-Local`（精确同参局部 3 行时间卷积/窗口）<br>• `P9C-NoHistory`（去除 A 帧历史消融）；<br>2. 严格单测：精确整数参数同构、`<1e-12` $\mathbb{C}^*$ 等变、零探针/未来/真值泄漏；<br>3. 3 颗预定种子开发集训练与端到端 CPU 延迟、峰值内存、每解对码块计算成本画像，落盘 `p9/results/p9c_dev_gate.json`。 |
| **CLI-4 (`P9-Audit`)** | 独立审计组 | `p9/independent_ref.py`<br>`p9/audit.py`<br>`p9/tests/test_p9_audit.py` | 1. 完全独立编写逐采样跨 A/B 边界连续卷积物理参考（覆盖机制 ①/②/③ 的边界复相位连续性校验）；<br>2. 独立微型码本完全穷举 ML 证书反演检验；<br>3. 独立盲评防泄漏防火墙（训练/测试轨迹交集、遮蔽探针投毒、隐藏机制标签、真实码字隔离、训练开销计数、无硬编码 `N0` 核验）；<br>4. 独立统计复算器（从逐链路原始 JSONL 复算汇总、按真实组数取自由度的 t 区间、精确 $2^G$ 符号翻转 Holm 家族与单侧 95% 非劣上界，以及单进程子组逐项重放）。 |

---

## 2. 汇合阶段：冻结协议、正式闭卷运行与论文级交付（由研究总负责人统筹）

四路并行 CLI 会话完成各自开发与单元测试后，由总负责人统一汇总验证：
1. **联合集成与端到端吞吐核算**：将 `p9/channel.py`、`p9/decoder.py`、`p9/gating.py`、`p9/neural_gate.py`、`p9/audit.py` 集成联调，依据实测每千块 CPU 开销确定正式组数（默认 `G = 16` 独立组 × `32` 条轨迹 = `512` 条轨迹 / `8,192` 共有码块，精确 `65,536` 符号翻转或按实测吞吐在冻结前锁定 `G=12..16`）。
2. **正式事前冻结 (`p9/PROTOCOL.md` + `p9/freeze.py` → `p9/results/freeze.json`)**：
   - 锁定训练/开发/正式同分布/正式分布外四套完全不相交种子池；
   - 锁定四项主 BLER 改善比较组成一个 **Holm-Bonferroni 家族（双侧 α=0.05，须方向符合主张）**：
     1. 强解码器相对原生方案（9 dB 真信道）；
     2. 经典历史门控相对开发集选定的最佳固定策略（少训练、三环境等权）；
     3. 所选注意力方案相对最强经典门控（少训练、三环境等权）；
     4. 所选注意力方案相对等参局部网络（少训练、三环境等权）；
   - 单列工程非劣性判据（少训练候选相对密集清空参考的共有 BLER 差值单侧 95% 配对上界 `< +1.00 pp`）及全帧净正确交付信息位。
3. **正式闭卷执行、独立审计复算与中文论文式交付 (`p9/P9_REPORT.md`)**：
   - 逐链路逐码块保存配对判决与证书记录，运行 `p9/audit.py` 独立复算 + 单进程子组重放；
   - 生成论文级多面板图表（SNR 错误率分解与 ML 下界证书、三机制+OOD 门控收益与连续链路代价、可靠性-净交付量-CPU 成本前沿），每项主张明确标注 **“支持 / 反证 / 证据不足”**，停在 P9 交由用户与 Codex 联合核实。
