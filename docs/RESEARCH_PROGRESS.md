# EM-FloatFormer 研究进展

更新于 2026 年 11 月 23 日。**当前已完成六阶段总计划的第 3 阶段（冻结编码器跨条件直读 ＋ `Ministral-3-3B-Instruct-2512-BF16` 首次真实 3B 接入全量评测），准备进入 Qwen3.5-4B 跨宿主复验与第 4 阶段实采衔接。** [六阶段总计划](LONG_TERM_RESEARCH_PLAN.md) · [宿主接入计划](MULTIMODAL_INTEGRATION_PLAN.md)

| 阶段 | 截至本次更新的交付与待验问题 |
|---|---|
| 1．可辨识边界 | 四相专题已交付。理想模型中相对相位恰需 2 比特，两问都保证答对恰需 4 比特。已推导离散噪声下的逐题风险、两种时间间隔的影响，以及连续 I/Q 下有限缓存的误差界。[证明与复算](https://github.com/20171806011/EM-FloatFormer/blob/fa396e8f5041d9db1f71af60d7b1558f1370d7fb/research/fourphase_memory_pilot/README.md) |
| 2．配对判别实验 | 首套 CPU 合成基准已交付。保持题面和逐点幅度相同，转动第二窗或两窗同转，分别揭示有效、已知独立、重复或缺省参考，逐项比较答案概率。[基准与可复跑样例](https://github.com/20171806011/EM-FloatFormer/blob/6462a6ae04c2d96a72b70d69b69e1ba21f36896e/research/fourphase_paired_questions/README.md) |
| 3．共享编码器与首次模型接入 | **已交付本阶段全量闭环**：① 继承 `504f3a44` 12 份冻结编码器与手工 4/8 比特表示完成 `R0` 回归与 `C1..C6` 跨支持域直接读出及留出第三窗 `Q3` 理论复核；② 冻结 `Ministral-3-3B-Instruct-2512-BF16`，仅训 `1.99M` 连接层于 Google Colab G4 Blackwell 完成 12 轮训练、开发集锁定及盲测对照。下一项为 **`Qwen3.5-4B` 跨宿主独立复验**。[阶段 3 实验报告](../research/stage3_ministral_cache/REPORT.md) · [理论推导](../research/stage3_ministral_cache/THEORY.md) · [冻结协议](https://github.com/20171806011/EM-FloatFormer/commit/8f6a134bc891d33aef81f9bbd5026493db1f332a) · [直读结果](../research/stage3_ministral_cache/results/direct/direct_summary.json) · [3B 宿主盲测](../research/stage3_ministral_cache/results/host/host_summary.json) · [中文追问实录](../research/stage3_ministral_cache/results/host/transcript_zh.md) |
| 4．独立采集验证 | 已有 Pluto、CTTC 与 435 MHz OFDM 接收研究。当前冻结编码器仍待用整场留出的真实采集验证其预测。[Pluto 报告](https://github.com/20171806011/EM-FloatFormer/blob/ade8333fb6812622083c631a336b4b69b69e1bcea/research/realdata_pluto_qpsk/REPORT.md) · [CTTC 报告](https://github.com/20171806011/EM-FloatFormer/blob/1887aee6de9649bc0dd8a405fe3c4542f3f6d77/research/realdata_cttc_qpsk/REPORT.md) |
| 5．事件与关系 | 既有相位跳变、候选解释与证据更新研究提供了可复用的物理问题。让同一表示在新波形上定位事件、报告歧义并指出支持时间窗，仍待验证。[前期五组会合](https://github.com/20171806011/EM-FloatFormer/blob/e914befb2c03838c5e71af60d7b1558f1370d7fb/research/abcde_convergence_2026_11_22/README.md) |
| 6．可追问的原生电磁模态 | 待交付多宿主真实录制演示与盲评。最终测试先编码、后揭题；能追加一次观测时，模型先选择参考窗并预测结果，再依据实际观测修订回答。 |

## 最近一轮改变了什么判断

1. **留出第三窗可辨识边界与反例证实**（[理论数值复核](../research/stage3_ministral_cache/verify_theory_output.json)）：仅保留 $U_1 = X_1+G$ 方向互信息的 4/8 比特相位保持缓存可在 `repeat_x1` 下突破均匀先验 $\log 4 = 1.3863$（2 dB 相干下 `hand_b8` 达 `66.19% / 0.9190 nats / NMSE=0.5854`，`independent_x3` 严格保持 `1/4 / 1.3863`）；而仅存相对相位的 2-bit 缓存虽能答对 Q1，却因 $I(U_1; Z)=0$ 在 Q3 上锁死于 `25% / 1.3863 nats`。迟到已知导频虽因 $G$ 对称性在揭示前四扇区严格等概率（`1/4`，偏差 `1.11e-16`），但在连续复平面 $\mathbb{C}$ 上呈现高度非均匀四叶玫瑰密度（`[0.1258, 0.1034, 0.1118, 0.00015]`），揭示后打破公共旋转盲区从而将 $X_1$ 从 `25%` 提升至 `64.5%–65.3%`。
2. **冻结表示跨支持域的真实代偿极限**（1,024 条 `s3_test` 母波形）：在离散/连续有限漂移（`C1..C3`）及 `0/6 dB` 跨 SNR（`C5`）下，8-bit 等变编码器与手工表示均通过 100% 旋转/保持/重复三项检验且准确率贴近完整 I/Q（`C1 64.75% vs 65.04%`，`C5-6dB Q1/Q3r` `89.36%/83.11%` 持平完整 I/Q）；但**2 dB 冻结表丢掉了连续亚扇区精细相位与真实 SNR 置信缩放**，导致在未见确定性频偏 **`C4 CFO`** 上 `Q1` 相对完整 I/Q 下滑 `-16.96 pp`（`95% CI [-20.31, -13.38]`），在 `C5` 出现 `8.5%–15.6%` ECE 校准失配；所有 `plain` 编码器仍在全部工况 `1,024/1,024` 失败三项对称检查。
3. **首次 `Ministral-3-3B`（3.43B 冻结宿主 ＋ `1.99M` 连接层）实测保留与损失**：在 `C1/C2/C3` 及未见 `C5 (0/6 dB)` 上，仅凭离散索引 `(z1, z2)` 与揭示参考，宿主即显著学会依据波形与导频修正信念（`C1 Q3r 64.58%` 与直读 `64.84%` 统计持平 `95% CI [-2.60, +2.08] pp`；`C5-6dB` `hand_b8` 达 `78.65% / 79.95% / 75.52%`），且 `text_only`（`25.0%`）与 `shuffled_Z`（`~25%`）对照证实预测严格依赖缓存向量而非文本先验。但宿主**无法自发对未见物理先验 `C4 CFO` 做解析补偿**（`Q3r 20.57%` vs 直读 `60.16%`，`Q2v 25.78%` vs `61.72%`）、**非等变连接层与自回归注意力破坏严格 $C_4$ 对称与无关参考隔离**（`rotate/hold` 失败 `44–115/128`）、且绕过前端 `dedup_references()` 的 `raw_duplicate_probe` 证明宿主自身会将重复 `<REF>` 令牌误放大（平均 TV `0.10–0.15`，置信增高可达 `+0.42`）。

## 第 3 阶段收口与第 4 阶段衔接

1. 本轮已在分支 [`research/stage3-third-window-ministral`](https://github.com/20171806011/EM-FloatFormer/tree/research/stage3-third-window-ministral) 交付完整代码、14 项单元测试、48 份 SHA-256 核验交付物及[阶段 3 实验报告](../research/stage3_ministral_cache/REPORT.md)。
2. 下一轮将在 **`Qwen3.5-4B`** 冻结宿主上独立复验上述连接层协议，并探索在输入端融入群等变/物理参数调制与真实采集数据对接。

总计划中的分支清单记录 11 月 22 日五组会合时的仓库。本页持续链接到固定提交与阶段报告，供后续进展对照。
