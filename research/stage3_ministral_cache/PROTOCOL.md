# EM-FloatFormer 第 3 阶段（Section B & C）事前冻结协议（Prospective Frozen Protocol）

> **协议冻结时间**：测试集评估前提交并推送（Commit `C1`）。任何基于新测试集反馈的参数调优、阈值更改或模型选择一律禁止。  
> **父研究分支与提交**：`research/fourphase-matched-cache` @ `504f3a440e737e290ba7e0fa62fd6667a04f5cde`  
> **路线图参考快照**：`main` @ `478ad59b23102622083c631a336b4bac19f1367f`  
> **本阶段新分支与目录**：`research/stage3-third-window-ministral` → `research/stage3_ministral_cache/`（绝对不修改 `main`、`research/matched_cache/`、`research/fourphase_paired_questions/`、`research/fourphase_memory_pilot/`）

---

## 1. 继承基线与 12 份冻结编码器指纹核验

所有 12 份学习权重与父基准脚本直接继承自 `504f3a44`，在本阶段**全程冻结参数与查找表，禁止重训**。加载 `.npz` 权重时强制 `allow_pickle=False`。重复导频去重逻辑由 [schema.py](schema.py) 与 [analytic.py](analytic.py) 的公共函数 `dedup_references()` 实现，与 `research/matched_cache/evaluate.py` 一致：按 `(acquisition_id, shares_channel == True)` 去重；若相同 `acquisition_id` 对应冲突样本向量则直接报错。

| 继承文件（`HEAD` canonical git blob） | 字节数 | SHA-256（LF 规范化 git blob） |
|---|---:|---|
| `research/fourphase_paired_questions/paired.py` | 9,704 | `05f700901f76e6cd8e82f71a921d6c4ac2efc657a7a40da1c1a61b3c24c744c8` |
| `research/fourphase_paired_questions/dev.json` | 3,341 | `c488195569e4402236d9be0d74a6884eabcb590ac62656e8cee66b737beb3724` |
| `research/fourphase_paired_questions/test.json` | 3,347 | `84959926654aec546048d6c9e61124dda91c01b6caf67ebff71810a678917140` |
| `research/matched_cache/results/summary.json` | 25,181 | `4fd2d3541257a5d8b0290d45af055f2a515261232425bd775e26aed74c75cdd0` |
| `research/matched_cache/weights/equivariant_b4_s17.npz` | 3,309 | `72b7f7ced01c26a59405cee73e4ae738c13a8b986f5ac8880b0ac47ac3044bee` |
| `research/matched_cache/weights/equivariant_b4_s29.npz` | 3,309 | `7274691608625ebf0bb4e1353b83e8347b091176044f2efef0daeb5c5b0ecdbb` |
| `research/matched_cache/weights/equivariant_b4_s43.npz` | 3,309 | `f8ede5120f9cbf9f96adb718090d5a6029e9390b0ea869cea2eaefbb20a987d9` |
| `research/matched_cache/weights/equivariant_b8_s17.npz` | 3,789 | `e67d9fbafb90791ab19549de7789798d818286d7f52d7ac566cc39df71e78356` |
| `research/matched_cache/weights/equivariant_b8_s29.npz` | 3,789 | `ee02ef91abefc300c89bbf658ade23ae45dc0eac8928a135668353ea5654c353` |
| `research/matched_cache/weights/equivariant_b8_s43.npz` | 3,789 | `f097fdecd8306f507df2087cad42cbdd9f7b34277f3c1f3283fc1a2bc0e82afb` |
| `research/matched_cache/weights/plain_b4_s17.npz` | 3,405 | `8b6042248c0301cbe39f7a1aa64676425ae8836906058cddc2313015280fa2b7` |
| `research/matched_cache/weights/plain_b4_s29.npz` | 3,405 | `fb8ef0ba44c8cea00cd2f5d089025597fe7813967fa1af61aa6af6f272ca0725` |
| `research/matched_cache/weights/plain_b4_s43.npz` | 3,405 | `2b9c71a5de73b2ceae64a9e01fabfc52efe4ee35435b018f2ba5d4ad8dc6c458` |
| `research/matched_cache/weights/plain_b8_s17.npz` | 4,173 | `d1857b226a496b5ce1013d28fde61944e45a42bf7afbf2361f953b38d130dd08` |
| `research/matched_cache/weights/plain_b8_s29.npz` | 4,173 | `efbd2f8026aa47dc992be816972412750e380284903dd7f06b9b1f38793185c7` |
| `research/matched_cache/weights/plain_b8_s43.npz` | 4,173 | `60a7b03efb1ffb49566f9f1489d79831d475ebf4ef7afa8815e6a99a9616a132` |

---

## 2. 严格互斥母波形划分、随机种子与支持范围

与父提交种子（`train=730241`, `dev=410061`, `test=518339`）严格不重叠，**同一条母波形的 3 个正交旋转派生版本 `(0,0)`, `(0,1)`, `(1,1)`（及坐标系同转）、两类第 3 窗协议（`repeat_x1`, `independent_x3`）及 5 种参考视图严格留在同一划分内**：

| 划分名称 | 随机种子 | 64-母本平衡块数 | 母波形总数 | 用途说明 |
|---|---:|---:|---:|---|
| `r0_dev` / `r0_test` | `410061` / `518339` | `4` / `16` | `256` / `1,024` | 原 2 dB 两问精确回归复算（断言与 `504f3a44` `summary.json` 偏差 $<10^{-12}$） |
| `s3_train` | `840101` | `16` | `1,024` | **仅用于** Section C 小连接层训练（绝不重训冻结波形编码器） |
| `s3_dev` | `840202` | `4` | `256` | Section C 早停、校准与锁定宿主容差 `host_dev_lock.json`（先于盲测锁定） |
| `s3_test` | `840303` | `16` | `1,024` | Section B 直接读出全部 1,024 母波形；Section C 3B 宿主依据 Colab 烟雾计时的预先锁定确定性子集（全量或前 4 块 256 母波形，于盲测前在 `host_dev_lock.json` 锁定） |

对全部划分运行 `orbit` 跨划分碰撞检测，断言任意两划分间原始三点复数窗轨道（$C_4$ 旋转等价类）交集严格为 **0**。

---

## 3. 训练支持范围 vs 测试新条件总表

12 份学习编码器（及 2 份手工表示 `hand_b4`, `hand_b8`）仅在 **`SNR=2 dB, h=0, 无漂移, 未旋转单窗`** 支持域上生成或拟合。在冻结后于下列 7 组条件跨支持域评测：

| 条件代号 | SNR (dB) | 窗间距 $(h, \ell)$ | 导频时刻 $t_p$ | 信道相位过程 | 训练支持内／外 | 重点检验机制 |
|---|---:|---|---|---|---:|---|
| `R0_reg_2db` | `2.0` | `(0, 0)` | `0` | 静止相干 | 支持内回归 | 复算 `504f3a44` 全部指标与检查 |
| `C1_coherent_2db` | `2.0` | `(0, 0)` | `0` | 静止相干 | 相干含 Q3+5 臂 | Q3 正例 vs 负对照；`duplicate_copy` vs `fresh_resample` |
| `C2_discrete_walk` | `2.0` | `(2, 2)` | `6`（晚于 W3） | 离散 $C_4$ 游走 $d=0.08$/步 | 未见离散漂移 | 分离 $h, \ell$ 衰落与晚导频；有限求和校验 |
| `C3_mid_pilot_cont` | `2.0` | `(4, 2)` | `2`（W1 与 W2 间） | 连续高斯游走 $\sigma_\phi=0.25$ rad/步 | 未见导频位置+连续 | 导频插入中间；完整 I/Q 分数角估计 vs 有限表 |
| `C4_cfo_ramp` | `2.0` | `(2, 2)` | `4`（W3 上方） | 确定性 CFO $\omega=0.35$ + $\sigma_\phi=0.10$ | 未见频偏坡道 | 检验对确定性先验轨迹的结合能力 |
| `C5_snr_0db_6db` | `0.0` / `6.0` | `(1, 2)` | `3` | 连续高斯 $\sigma_\phi=0.20$ rad/步 | 未见 SNR 失配 | 冻结 `2 dB` 编码表外推（对比名义表 vs 校准核） |
| `C6_decoherent_order` | `2.0` | `(2, 2)` | `4` | 强失相干 $\sigma_\phi=3.0$ rad/步 | 失相干极限+新题序 | 检验崩溃回 `1/4 / log 4`；先 Q3 后 Q1，及导频后重答 Q3 |

注：Section C 训练仅使用 `C1_coherent_2db`, `C2_discrete_walk`, `C3_mid_pilot_cont` 三种基础训练工况；**`C4_cfo_ramp`、`C5_snr_0db_6db`、`C6_decoherent_order` 对 3B 连接层均属严格未见（Out-of-Support / Unseen Question Order）条件！**

---

## 4. 五类参考视图与重复样本去重契约

每条母波形 × 旋转变体均独立评分以下 5 个参考臂（确保同一正样本新采样与同一向量精确副本严格分开）：
1. `none`：无任何参考 `()`；
2. `valid`：1 个同信道导频 `(Reference(acquisition_id=0, shares_channel=True, samples=P_1),)`；
3. `independent`：1 个异信道独立导频 `(Reference(acquisition_id=0, shares_channel=False, samples=P_indep),)`；
4. `duplicate_copy`：同一物理采集含噪向量的完全重复副本 `(Reference(0, True, P_1), Reference(0, True, P_1))`；
5. `fresh_resample`：同一导频符号 $s_0=1$、经过新独立接收噪声（含各自时刻信道）采样的第 2 条同信道参考 `(Reference(0, True, P_1), Reference(1, True, P_2))`。

**去重边界契约**：
- 直接读出器：`dedup_references()` 在编码与查表之外按 `(acquisition_id, shares_channel)` 去重（对 `duplicate_copy` 必与 `valid` 逐比特相同；对 `fresh_resample` 合并两次独立似然）；另设故意不去重的 `double_count_control` 负对照。
- 3B 语言宿主：默认通过同一 `dedup_references()` 预处理输入保证参考权限与直接读出**绝对一致**；另设 **`raw_duplicate_probe`** 对照——绕过前端去重直接把两份带同 `acquisition_id=0` 文本标记的参考槽位喂给宿主，测量宿主注意力自身能否抑制或仍会误放大重复证据的置信度！

---

## 5. 题面、候选映射与旋转预期变换

所有方法（直接读出与 3B 宿主）在完全相同的中文题面下作四选一概率输出（最低下标打破平局）：
- **Q1**：`第二个窗口的发射符号相对第一个窗口转了几个四分之一圈？` → `["0圈（0°）", "1圈（+90°）", "2圈（180°）", "3圈（+270°）"]`
- **Q2**：`第一个窗口原来发送的是哪个四相符号？` → `["符号0（1）", "符号1（i）", "符号2（-1）", "符号3（-i）"]`
- **Q3（揭示前扇区预测）**：
  - 正例 `repeat_x1`：`协议规定第三个窗口重发第一个窗口的符号，且已知当前时间间隔。在揭示第三个窗口观测前，其接收相位最可能落入哪个四分扇区？` → `["扇区0（-45°到+45°）", "扇区1（+45°到+135°）", "扇区2（+135°到-135°）", "扇区3（-135°到-45°）"]`
  - 负对照 `independent_x3`：`协议规定第三个窗口发送与前两窗独立均匀的新四相符号。在揭示第三个窗口观测前，其接收相位最可能落入哪个四分扇区？` → 同上四扇区。

按母波形在三种窗口干预 `(0,0)`, `(0,1)`, `(1,1)` 与全坐标旋转 `(1,1,1,1)` 下的严格概率检验条目（直接读出数值容差 $\tau_{\text{direct}} = 3 \times 10^{-13}$）：
- `(0,1)`：Q1 顺次 `roll(+1)`；Q2 **保持**；Q3（`repeat_x1` 与 `independent_x3` 均）**保持**。
- `(1,1)`：Q1 **保持**；Q2 `roll(+1)`；Q3 `repeat_x1` `roll(+1)`（复均值 $\times i$），Q3 `independent_x3` **保持**。
- `duplicate_copy` vs `valid`：Q2 四维概率差 `max_abs <= tau` 且 `max_confidence_increase == 0.0`；`fresh_resample` vs `valid`：平均 Q2 NLL 严格下降。

---

## 6. 模型对比系列、评估指标与 3B 宿主盲测锁定规则

1. **并行评估方法**：
   - `prior_no_waveform`（不看波形仅看题面先验，其中 Q1/Q2/Q3 全均匀，作为底线参照）
   - `full_iq_known_param`（完整复数 I/Q 观测 + 已知漂移参数精确解析条件最优解 —— **单列注明已知真实漂移参数**）
   - `hand_b4`, `hand_b8`（同预算手工保相表示）
   - `plain_b{4,8}_s{17,29,43}`（普通学习编码器，6 份权重逐一报告与聚合）
   - `equivariant_b{4,8}_s{17,29,43}`（旋转约束学习编码器，6 份权重逐一报告与聚合）
   - `ministral_3b_host`（冻结 `mistralai/Ministral-3-3B-Instruct-2512-BF16` 语言主干 + 冻结相同波形编码器，仅训练小型投影连接层；配对对比 `equivariant_b8_s17`、`hand_b8`、`plain_b8_s17`）
   - 宿主三大消融对照：`text_only`（剥离波形向量仅余文本与元数据）、`shuffled_Z`（仅在完全相同条件与题面内跨母波形随机置换 $Z$）、`raw_duplicate_probe`。
2. **统一指标**：
   - 逐题准确率、自然对数损失（NLL，nats）、ECE（10 分箱）、Brier 分数；Q3 额外报告揭示前条件均值复数 **NMSE**。
   - 配对差异：以**母波形**为聚类单元，跨母波形变体取平均后采用 2,000 次确定性有放回自助法（seed `17429`）计算点估计与 **95% 置信区间**。
   - 严格保存每种代表性模型与宿主在测试集上表现最差的**可重放失败样例**（母波形 ID、条件、输入、预期 vs 实际预测概率）。
3. **宿主容差开发集锁定（`host_dev_lock.json`）**：
   - 因 BF16/FP16 与 3B 自回归前向存在有限精度波动，宿主概率一致容差 $\tau_{\text{host}}$（如各检验 `p95`／上限阈值与温度缩放标量 $T$）必须在 Google Colab GPU 的 `s3_dev` 上于接触 `s3_test` **之前**测量并写入 `host_dev_lock.json` 锁定；盲测时严格按该锁定阈值判别宿主旋转／保持／重复三项通过情况。
4. **执行顺序与硬边界**：
   - 本阶段仅接入并完成 **Ministral-3-3B-Instruct-2512-BF16** 首次 3B 实跑与全量对照；`Qwen3.5-4B` 跨宿主复验严格留待下一轮。
   - 3B 模型加载、连接层训练、盲测评分与中文追问生成全部在 **Google Colab GPU** 上运行，禁止在本地下载或加载 3B 基础权重；仅取回轻量化连接层权重（`< 5 MB`）、逐项汇总结果、最差样例、中文多轮记录与 `MANIFEST.sha256` 校验清单。
