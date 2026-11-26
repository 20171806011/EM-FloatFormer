# Stage 4 第一性原理数学与权重微调全量综合报告（10 轨协同收敛定稿）

> **仓库基线**：`387e7bc749150a69fb317b3f1351ad0d07b3bfad` ｜ **目标分支**：`research/stage4-math-and-realdata`  
> **双路多智能体协作体系**：
> - **系统与协议数学组（`M1–M6`）**：OpenAI Codex CLI（`gpt-6.2-sol`，Round 1 独立推导 + Round 2 跨轨质询），全部完整对话与 JSONL 事件流归档于 [`discussions/M1..M6/`](discussions/M1/FULL_CONVERSATION.md)  
> - **权重修改与机制优化组（`W1–W4`）**：原版 Antigravity CLI（`agy-bin.exe` v1.5.13，`--model gemini-4-neon-xhigh --effort xhigh`，Round 1 + Round 2 交叉对齐），全部完整对话归档于 [`discussions/W1..W4/`](discussions/W1/FULL_CONVERSATION.md)  
> **全量独立复现总控**：[`verify_all.py`](verify_all.py) 已在无竞争 CPU 环境下顺序完成 10 个独立验证脚本、10 轨完整对话及仓库卫生闸门校验，总耗时 **`825.702 s`**，**`10/10` 全部退出码 `0`**，详见 [`verify_all_summary.json`](verify_all_summary.json)（`all_passed: true`、`scripts_passed: true`、`discussions_passed: true`、`hygiene_passed: true`、`protected_trees_untouched: true`、`Spend: $0.00`）。

---

## 一、第一性原理统一架构：零试错三层决策金字塔

在正式消耗任何 Modal A100-80GB 算力前，10 条独立研究轨道围绕“**如何以最低试错成本、严格零通用能力回退，在 `Qwen3.5-4B / 9B / Qwen3.8-27B` 上打通长上下文物理推演、真实 SDR 数据外推与 LEMMO 公平归因**”展开两轮对抗与收敛，最终形成如下不可逾越的三层工程决策金字塔（完整可执行规则见 [`STAGE4_PROTOCOL.md`](STAGE4_PROTOCOL.md)）：

### Layer 0：冻结宿主 + 无损连接器 + 零参数记忆旁路（`0` 宿主权重改动，`$0` 额外 GPU）
第一性原理下，凡是仅依靠**输入表示重构、精确充分统计量缓存与前向读出几何规范化**即可无损解决的问题，一律禁止提前解锁骨干网络权重：
1. **严格因果真实 SDR 前端与防泄漏切分（[`M4`](M4/THEOREM_M4.md)）**：
   - Pluto (`QPSKlink.mat`, `6,240` 有效帧：Train `3,000` / Dev `1,000` / Test `2,240`，划分为 `16` 个独立测试簇×`140` 帧，每边界严格设置 `100` 帧保护带；Pluto `refs=[15, 18]`，应力样本 `980` 行）与 CTTC (`train/dev/test = 1,440/864/1,440` 目标包 + `576` 参考包，`72` 测试会话环簇，`minimum_acquisition_gap_samples_after_rcv_support_expansion = 16,481`，每分段 `4,096` 循环保护，`f708_snr10` 独立作为跨频段/SNR 留出防线，`refs=[126, 127]`）。
   - 双缓存上限代理显示 `full16` 相对 `s17` 的联合 NLL 理论可回收增益达 **`0.29424–0.34800 nats`**（符号 NLL 增益 `0.00007–0.06130 nats`）。
2. **16-bit 无损双缓存线协议 `m5-c16-v1-le`（[`M5`](M5/THEOREM_M5.md)）**：
   - 位宽分配 `[2b turn | 2b s17_inner | 6b angle | 4b log_radial | 2b quality]`，保留离散 `s17` 比特级无损重构的同时保留复径向极坐标信息（消除原 `s17` `0.4804 TV` 径向碰撞），在 Stage 6 主动检索合同 `m5-stage6-blind-v1` 下支持单次读取决策。
3. **字节级幂等参考池 `RefSufficientPool`（[`M3`](M3/THEOREM_M3.md) · [`ref_pool.py`](M3/ref_pool.py)）**：
   - 在联合观测协方差下，通过块矩阵 Schur 补精确计算新参考观测的条件新息（与全高斯联合对数似然误差 **`7.99×10^-15`**，后验误差 **`8.88×10^-15`**）。
   - 对重复副本、无关干扰项、列表重排与填充实现**字节级恒等前缀**与 **`max_KL = 0.0`、`max_TV = 0.0`**（有效参考提升至 `66.60% / 0.8193 nats`，新增独立重采样升至 `74.22% / 0.6629 nats`）。
4. **`C_4` 商群等变连接器（[`M1`](M1/THEOREM_M1.md)）**：
   - 仅训练输入侧等变连接器，在冻结宿主打分下即在双外推象限跨越 `55%` 门槛：**`Host-Pure` 达 `G1 = 64.60%`（NLL `0.9785`）、`G3 = 61.91%`（NLL `1.0130`）**，且全任务 `C_4` 等变缺陷 **`0.0`**（`Physical+Residual` 达 `G1 = 64.62% / 0.9422`、`G3 = 62.11% / 0.9810`）。
5. **零参数 `W1+W3` 双轨零空间慢头旁路（[`W3`](weight_mod/W3/THEOREM_W3.md) · [`W1`](weight_mod/W1/THEOREM_W1.md)）**：
   - 当不解冻早期层 `1–10` 时，基于 Task-A Fisher 能量最低慢头施加零新增可训练参数的影轨（`α_slow` 温启动 + 严格 `m_EM(t)` 零散文写入 + `P_{⊥,C}` 读出过滤），在 `L_gap = 32` 上将早期 GDN 保持率提升 **`77,919×`**（`7e-6 → 0.779193`），同时保持 **`PureKL = 0.000`** 与 **`MixKL = 1.172e-3 nats`**（彻底消除 `leaky_beta` 导致的 `1.256e-1 nats` 散文污染灾难）。

### Layer 1：Two-Probe Phase-A VoI 闸门与微创权重修改（`<= 2.20M` 对齐 LEMMO Tier-P1）
并非所有场景都需要修改 LLM 权重。**[`W4`](weight_mod/W4/THEOREM_W4.md)** 与 **[`W1`](weight_mod/W1/THEOREM_W1.md)** 联合推导出 **Two-Probe Phase-A 信息价值（VoI）分流定律**：
- **探针 A（连接器 DPI 检验）**：在冻结 Phase-A 表示 $E_{\theta_A}(x)$ 上拟合轻量线性/MLP 探针；若探针与最优贝叶斯后验仍有 `> 0.030 nats` 差距，立即在 CPU 修复连接器（**`$0` 额外 GPU 浪费**）。
- **冻结阈值**：若探针 A 达标（`<= 0.030 nats`）且冻结宿主相对真实后验的路由缺口 `Δ_host <= 0.030 nats`，**严格维持冻结宿主**，执行 **Branch 0（`$8.3906` 不含重试 / `$9.0708` 含 `900s` 重试 / `$9.50` 硬顶）** + 完整 384 步 Tier-P1 LEMMO。
- **微创解锁阈值**：仅当 `Probe A <= 0.030 nats` 且 `Δ_host ∈ [0.040, 0.400] nats`（若 `Δ_host > 0.315 nats` 且 `σ_3(D_answer) < 0.10`，先施加 `W1` 零参数 `dt_bias` warm-start 将 $\sigma_3$ 抬升 `1.73×` 扩大 VoI 窗口至 `[0.04, 0.40]` nats）时，触发 **Mode I (`Stop-Grad L10`) Surgical-96** 微创修改：
  1. **广义 Rayleigh 分层与 Mode-I `stop_grad` 显存账本（`W3` + `W4`）**：仅解冻第 `10` 层之后的黄金混合带（4B/9B：**`FA [12, 16, 20, 24, 28]` + `DN [11, 15, 19, 23]`**；27B：**`FA [20, 24, 28, 32, 36, 40, 48, 56]` + `DN [19, 23, 27, 31, 35, 39]`**）。仅当严格冻结 `Phase A` 连接器并在第 `10` 层执行 `stop_grad`（`Option B1 / Mode I`）时，才真正省下 **`11.827 GiB`** 反向传播激活显存（4B 峰值 **`40.028 GiB`**、`2,260.0 ms/step`；若错误采用 `Option B2` 共调连接器则第 1–10 层激活节省 **`0.0 GiB`**，飙升至 `51.855 GiB`）；此举同时将 `Qwen3.5-9B`（`mb=16`）峰值显存从 `67.270 GiB` 拉回至 **`53.426 GiB <= 65 GiB`** 安全红线内。
  2. **复对称化零空间 Cayley + Schur-Irrep（`W2` 定理 7/8 + `W3` 定理 9）**：通过酉复化平均 $C_{\mathrm{gen}}^{\mathbb{C}} = \frac{1}{2}(C_{\mathrm{gen}} + J_d^\top C_{\mathrm{gen}} J_d)$ 构造复零空间投影 $P_{\perp, C}^{\mathbb{C}}$，确保 $[P_{\perp, C}^{\mathbb{C}}, J_d] = 0$、`U(m)` 严格酉正交（`6.66e-16`）、$SO(2)$ 对易（`1.11e-16`）并在通用语言前 $2k_c$ 维主旋转子空间保持严格恒等映射（**`860×–3,285×`** 文本扰动能量抑制）；**Cayley $8\times 8$ Woodbury 核心必须锁定 `FP32`**（若降为 4-bit NF4，正交缺陷将放大 **`20,162.1×`**！）。配合 `W2` 规范对齐框架 $B_{\mathrm{align}} = M_{\mathrm{cand}}^+ F_{\mathrm{real}}$ 与商群包装（`r>=3`、`C4=0.0` 且 **`λ_orbit = 0`**），不仅节省一次 `rot1` 前向并使跨种子 NLL 标准差崩塌 **`27.7×–40.6×`**，在 `W1` 全栈联动实验中更于 `G1` 达 **`64.36% / 0.9715`**、未知半圈 `G5(θ=π)` 达 **`63.28% / 1.0118`**（所有无 Schur 方案仅为 `5.71%–17.77%`）、且在未见超长间隔 `OOD@128` 实现 **`0` 衰减（`64.36% / 63.28%`）**！
  3. **双层防遗忘门控实现 `MixKL <= 1.0e-3 nats`（`W3` + `W1`）**：序列级二进制门 $g_{\mathrm{EM}} \in \{0, 1\}$ 锁死离题通用任务 **`PureKL = 0.000e+00`**；Token 级门 $m_{\mathrm{EM}}(t)$ **仅遮蔽 K/V/β 写路径，严禁遮蔽 Q 读出**，使在轨混合对话 `MixKL` 于 `W3` 压降至 **`8.984e-4 nats <= 1.0e-3`**（相对无约束 LoRA `1.072 nats` 下降 **`1,193×`**）。
  4. **三档参数规模均严格位于 `<= 2.20M` Tier-P1 公平归因红线内**：
     - **`Mode-I 精简核心`（`W3/W4` 默认基准）**：4B 总计 **`1,486,878`** 参数（M1 `488,400` + 宿主适配 `998,478`，占 `2,184,964` 上限 **`68.05%`**）；
     - **`头内复 Cayley 全对易版`（`W2` 完整收敛）**：4B 总计 **`1,619,678`** 参数（`488,400` + `1,131,278`，占 `74.1%`）；
     - **`Tier-1 全门控增强版`（`W1` 可选扩展）**：4B 核心 **`1,974,782`** / 含 `3,7` 早期 GDN 门控例外 **`2,015,758`** / 含 Step-69 触发双层稀疏 SwiGLU **`2,116,094`** — 全部严格低于 `2,184,964` 基线。
  5. **两阶段反冲击 + Rao–Blackwell 蒸馏 + 无规范合并（`W4` + `W3` + `W2`）**：两阶段课程消除冷启动 `3.604×` Step-0 梯度冲击与 `9.732×` 遗忘漂移；用闭合后验 KL 替代 1-hot CE（W1/W2/W3 梯度方差实测分别/最高下降 **`90.80×`**、`16.94×`、`113.73×`），使 **96 步（`Surgical-96`）即达 160 步收敛率的 `99.3%` 且已满足 `MixKL = 9.002e-4 <= 1e-3`**；多频域/SNR 适配器融合时采用 **规范 Lie+Schur 凸均值合并**（消除 LoRA `61.41%` GL(r) 规范误差，将 Pluto↔CTTC 交叉干扰 $\Gamma_{21}^{\mathrm{soup}}$ 砍半 **`16.9×`** 至 `0.1382`，未知混合捕获达 **`73.83% / 0.7904`**）。

### Layer 2：LEMMO 公平归因、`9B/27B` 扩展律与 Stage 6 主动盲测（全流程严格 `<= $12.88`）
1. **精确词表与宿主拓扑校验（`|V| = 248,320`，四轨 1:1 统一）**：
   - `Qwen3.5-4B`：`4,205,751,296`（`tie_word_embeddings=True`）；`Qwen3.5-9B`：`8,953,803,264`（`False`）；`Qwen3.8-27B`：`26,895,998,464`（`False`）。
   - **量化禁区定理（`W4` Exp 2B）**：对 Gated-DeltaNet 门控做 NF4 量化会导致相位重构 MSE 暴涨 **`3,065.8×`**、NLL 崩溃 **`+0.72827 nats`**；因此 27B 大模型强制执行 **Mixed `MLP-NF4` + `DeltaNet/FullAttn BF16` + `Cayley FP32`**（KL 漂移仅 **`0.00032 nats`**，峰值显存 **`45.267 GiB`**）。
2. **`$12.88` 零透支严格十进制账本（`W4` + `M6`）**：
   - 停留在冻结宿主（Branch 0）：`$8.3906`（含 `900s` 重试 `$9.0708`，硬顶 `$9.50`）+ 完整 Tier-P1 LEMMO `$10.5067`（含重试 `$11.3682`）；
   - 解锁 Mode-I `Surgical-96`（`$10.0102` / 含重试 `$10.9170`）+ 自适应 Tier-P1 LEMMO **`Option L2`**（384 步×`20,480` 公共支持样本：**`$11.4740`** / 含 `1,440s` 重试 **`$12.5622`**）或 **`Option L3`**（384 步×全量 `44,812` 盲测样本共享 `900s` 重试：**`$12.1262`** / 含重试 **`$12.8063`**）。**10,000 次蒙特卡洛模拟显示 `100.0%`（max `$12.8508`）严格 `<= $12.88`，`Billed Cost = $0.00`！**
3. **Stage 6 `m5-stage6-blind-v1` 主动 `max_EIG` 盲测（`M5`）**：在三类退化信道族（`jump_interval`、`selective_fade`、`cfo_alias`）各取 1 次补充窗口读取预算下，主动 `max_EIG` 达到 **NLL `0.04823` / 准确率 `98.20%`**，远超开发集固定最优 `fixed_best_on_dev`（`0.51010 / 66.23%`）及回顾神谕固定最优（`0.50672 / 66.38%`），`30` 项配对 Bonferroni 检验（临界值 `3.14607`）全部显著通过。

---

## 二、10 轨研究全量索引表（指标、无竞争实测耗时与文件链接）

> 下表中 CPU 复现耗时**全部**直接引自 [`verify_all_summary.json`](verify_all_summary.json) 对 10 个独立脚本的顺序实测记录（总计 `825.702 s`，`10/10` 退出码 `0`）。

| 轨道 | 协作主体 / 模型 | 核心数学与系统贡献 | 关键量化验证指标 | 顺序耗时 (`verify_all`) | 定理文档 / 脚本 / 结果 / 完整对话 |
|---|---|---|---|---:|---|
| **M1** | Codex CLI (`gpt-6.2-sol`) | $C_4$ 商群等变连接器、Jacobian 可达球审计与固定文本槽位干预 | `Host-Pure`：`G1 64.60%`、`G3 61.91%`，全任务 `C4` 缺陷 `0.0`；槽位对调使 `G1` NLL 恶化 `2.0172 nats` | `154.226 s` | [定理](M1/THEOREM_M1.md) · [脚本](M1/verify_m1.py) · [JSON](M1/results_m1.json) · [对话](discussions/M1/FULL_CONVERSATION.md) |
| **M2** | Codex CLI (`gpt-6.2-sol`) | Bregman/KL 取证检验、IUT+Holm 族系控制与零 α 独立 Dev 预筛 | Stage3D 最大复现差 `0.0`；`M6` 阈值下条件假阳性 `1.667%`（MC `1.629%`）、联合 `1.011%`（`0.988%`） | `18.083 s` | [定理](M2/THEOREM_M2.md) · [脚本](M2/verify_m2.py) · [JSON](M2/results_m2.json) · [对话](discussions/M2/FULL_CONVERSATION.md) |
| **M3** | Codex CLI (`gpt-6.2-sol`) | 最小贝叶斯充分池 `RefSufficientPool`、相关噪声 Schur 补与共享 CFO 两时钟模型 | Schur 似然差 `7.99e-15`；重复/干扰项/重排 `KL=TV=0.0`；两时钟 `0.1477` vs 合并池 `0.3784` NLL | `14.781 s` | [定理](M3/THEOREM_M3.md) · [实现](M3/ref_pool.py) · [脚本](M3/verify_m3.py) · [JSON](M3/results_m3.json) · [对话](discussions/M3/FULL_CONVERSATION.md) |
| **M4** | Codex CLI (`gpt-6.2-sol`) | Pluto (`6,240` 帧) & CTTC 严格因果窗口切分、`f708_snr10` 留出、双缓存上限与 A100 账本 | 6 类信道双缓存相对 `s17` 联合增益 **`0.2942–0.3480 nats`**；Pluto/CTTC 无泄漏隙口 `>= 16,481` 样本 | `15.151 s` | [定理](M4/THEOREM_M4.md) · [脚本](M4/verify_m4.py) · [JSON](M4/results_m4.json) · [对话](discussions/M4/FULL_CONVERSATION.md) |
| **M5** | Codex CLI (`gpt-6.2-sol`) | 16-bit 双缓存 `m5-c16-v1-le` 与 Stage 6 主动条件 EIG 单次读取合同 `m5-stage6-blind-v1` | `s17` 消除 `0.4804 TV` 碰撞；`max_EIG` **`98.20% / 0.04823 nats`** vs `fixed_best_on_dev 66.23% / 0.51010` | `241.400 s` | [定理](M5/THEOREM_M5.md) · [脚本](M5/verify_m5.py) · [JSON](M5/results_m5.json) · [对话](discussions/M5/FULL_CONVERSATION.md) |
| **M6** | Codex CLI (`gpt-6.2-sol`) | 3-stage 序贯 VoI DAG、102 项 Stage3D 校验与 A100-80GB `$9.50` 安全预算 | 102 项最大误差 **`3.56e-16`**；序贯预筛省本 **`59.98%`**、单终点效能 **`94.04%`**、边界虚无 **`0.733%`** | `6.191 s` | [定理](M6/THEOREM_M6.md) · [脚本](M6/verify_m6.py) · [JSON](M6/results_m6.json) · [对话](discussions/M6/FULL_CONVERSATION.md) |
| **W1** | Antigravity 原版 (`gemini-4-neon-xhigh`, xhigh) | GDN↔Kalman-RLS (`2.22e-16`)、8-神经元 SwiGLU 旋转、L1–10 垂直存活与 $\sigma_3 \to \Delta_{\text{route,max}}$ | `unified` 全栈在 `L_gap=64` 达 **`64.36% G1 / 63.28% G5 / Q2 +8.85 pp`**，`OOD@128` 保持 `1.000`，`PureKL = 0.0` | `99.863 s` | [定理](weight_mod/W1/THEOREM_W1.md) · [脚本](weight_mod/W1/verify_w1.py) · [JSON](weight_mod/W1/results_w1.json) · [对话](discussions/W1/FULL_CONVERSATION.md) |
| **W2** | Antigravity 原版 (`gemini-4-neon-xhigh`, xhigh) | 复对称化零空间 Cayley（定理 7）、规范基 Lie+Schur 合并（定理 8）与 `λ_orbit=0` | `4d` `MixKL` 砍 **`27.6×–117×`**（`0.00799`）、Step-69 NLL SD 降 **`40.6×`**；Lie+Schur 合并 $\Gamma_{21}^{\mathrm{soup}}$ 降 **`16.9×`** | `85.300 s` | [定理](weight_mod/W2/THEOREM_W2.md) · [脚本](weight_mod/W2/verify_w2.py) · [JSON](weight_mod/W2/results_w2.json) · [对话](discussions/W2/FULL_CONVERSATION.md) |
| **W3** | Antigravity 原版 (`gemini-4-neon-xhigh`, xhigh) | 两层门控 `m_EM(t)`、`Option B1/B2` 诚实界定、零参数慢头旁路与 `Step-96` 截断证明 | `MixKL = 8.984e-4 <= 1e-3`（降 `1,193×`），`PureKL = 0.0`，慢头保持提升 **`77,919×`**，`Step-96` 达 160 步 **`99.3%`** | `130.985 s` | [定理](weight_mod/W3/THEOREM_W3.md) · [脚本](weight_mod/W3/verify_w3.py) · [JSON](weight_mod/W3/results_w3.json) · [对话](discussions/W3/FULL_CONVERSATION.md) |
| **W4** | Antigravity 原版 (`gemini-4-neon-xhigh`, xhigh) | `\|V\|=248,320` 精确显存与 NF4 陷阱（`3,065.8×`/`20,162×`）、两阶段反冲击与 `$12.88` VoI+LEMMO 账本 | Stage3D 校准误差 **`0.332%`**；4B Mode I **`40.028 GiB / 2,260 ms`**；10k MC `max=$12.8508 <= $12.88`（**`100.0%`**） | `54.115 s` | [定理](weight_mod/W4/THEOREM_W4.md) · [脚本](weight_mod/W4/verify_w4.py) · [JSON](weight_mod/W4/results_w4.json) · [对话](discussions/W4/FULL_CONVERSATION.md) |

---

## 三、Round 1 → Round 2 跨轨核心张力收敛矩阵

两轮协作最重要价值在于**通过第一性原理数学反驳，在零 GPU 开销下消除了 6 项潜在致命工程陷阱**：

| 跨轨张力（发起方 ↔ 冲突方） | Round 1 潜在隐患 | Round 2 数学收敛与工程终局结论 |
|---|---|---|
| **1. 早层冻结 vs 长间隔遗忘**<br>（`W3` 冻结 L1–10 ↔ `W1` 早层 GDN $\bar{\alpha}^{64}=0.0068$ 衰减） | 若第 1–10 层冻结且未区分垂直跳过与水平衰减，要么被迫全解冻多花显存、要么 naive `dt_bias` 泄露导致 `MixKL` 暴涨 **`90×`**（`0.1256 nats`）。 | **两级兼容解**：默认依靠 Pre-Norm 垂直残差直通第 11 层并启用 `W1+W3` 零参数慢头旁路（`77,919×` 保留率、`MixKL=1.172e-3`）；若实车 A100 探针显示早层垂直奇异值 $\sigma_3 < 0.05\sigma_1$，仅对 L3/7 慢头门控微解冻（`4B: +40,976` 至 `+82,944` 参数），依然严守 `<= 2.20M`。 |
| **2. 实 PCA 零空间破坏 $SO(2)$**<br>（`W3` 实 PCA $P_{\perp, C}$ ↔ `W2` 复 Cayley 与 $[Q, J_d]=0$） | 在实激活协方差上直接前后投影会使 $[P_{\perp,C}, J_d]=0.2832$（W3 8 层代理达 `3.376`），直接破坏 `Q` 的正交性（$\kappa_2 = 2139.8$）或 $SO(2)$ 对易性（`0.441`）。 | **复对称化定理（`W2` 定理 7 / `W3` 定理 9）**：先做 Haar 平均 $C_{\mathrm{gen}}^{\mathbb{C}} = \frac{1}{2}(C_{\mathrm{gen}} + J_d^\top C_{\mathrm{gen}} J_d)$ 再将倾斜埃尔米特生成元限制入 $\Pi_\perp \mathfrak{u}(m)\Pi_\perp$。实测 $[P_{\perp,C}^{\mathbb{C}}, J_d] = 0.0$、因子-2 Loewner 谱上界严格成立（$\lambda_{\max} = -7.01\times 10^{-4}$）、通用文本 KL/能量抑制达 **`730×–3,285×`**，同时保住 4B **80% 复二重态带宽**（1,024 对）。 |
| **3. 跨硬件域合并规范塌陷**<br>（`W4` 命题 4.1 GL(r) ↔ `W2` 多域适配） | 普通 LoRA 因子平均存在 `GL(r)` 可逆矩阵自由度，导致 **`61.41%` 规范误差**；即使用权重汤，Pluto↔CTTC 频偏/抖动交叉干扰 $\Gamma_{21}^{\mathrm{soup}}$ 高达 **`2.338`**，`G5(θ=π)` 仅 `11.69%`。 | **规范基 Lie+Schur 合并（`W2` 定理 8）**：$B_{\mathrm{align}}$ 将左稳定子缩为 $\{I_3\}$（因子平均误差 `8.94e-8`），Lie 代数线性凸合并保证精确 `U(m)` 闭合（`1.33e-15`），$\Gamma_{21}^{\mathrm{soup}}$ 骤降 **`16.9×`（`0.1382`）**，未见混合域达 **`73.83% / 0.7904`**、未知 `G5(θ=π)` 达 **`61.33% / 1.0004`**。 |
| **4. Stop-Grad 虚幻节省、`|V|` 口径与 `96` 步收敛**<br>（`W4` ↔ `W3`） | Round 1 `W3` 误用旧词表 `151,936` 且未区分连接器冻结与共调的显存差异；`W4` 担心 `96` 步会牺牲 `MixKL`。 | **1:1 账本统一**：三模型均锁定 **`\|V\|=248,320`**；明确 **`Option B1 / Mode I`（冻结连接器+第10层截断梯度）** 是节省 `11.83 GiB`（4B）/`13.84 GiB`（9B）与 `-19.82%` 步耗的**唯一条件**；同时 `W3` 实测证实在商群 `λ_orbit=0` 下 **Step-96 已获 Step-160 收敛 `99.3%` 且 `MixKL = 9.002e-4 <= 1e-3`**，解锁 `Surgical-96 + LEMMO L2/L3` 全部落入 `$12.88`。 |
| **5. NF4 量化边界与 Cayley 精度**<br>（`W4` ↔ `W2/W1`） | 27B 显存吃紧时容易误将 QLoRA-NF4 一刀切套用到全部线性层及 Cayley 适配器。 | **精度红线**：DeltaNet NF4 使相位 MSE 恶化 **`3,065.8×`**（NLL `+0.728 nats`），Cayley 4-bit 化使正交缺陷放大 **`20,162.1×`**。因此 27B 强制执行 **Mixed `MLP-NF4` + `DeltaNet/FullAttn BF16` + `FP32 Cayley`**（`45.267 GiB`）。 |
| **6. 盲测统计控制与双缓存协同**<br>（`M1–M6` 内部收敛） | 同一批 dev 分数重复用于早停与最终检验会使 Cond. Size 膨胀至 `2.75%–8.24%`；旧参考池重放会导致方差低估或顺序偏移。 | 锁定 **仅 Dev 独立预筛（零 α 消耗） + IUT/Holm**；`RefSufficientPool` 确保精确充分统计量增量无双计；`m5-c16-v1-le` 16-bit 报文支撑 Stage 6 主动 `max_EIG`（较固定最优提升 **`+31.97 pp`** 准确率）。 |

---

## 四、模型可扩展性、`4B → 9B → 27B` 硬件参数表（`|V| = 248,320`）

下表综合 [`W4/results_w4.json`](weight_mod/W4/results_w4.json) 与 [`W2/THEOREM_W2.md`](weight_mod/W2/THEOREM_W2.md) / [`W1/THEOREM_W1.md`](weight_mod/W1/THEOREM_W1.md) / [`W3/THEOREM_W3.md`](weight_mod/W3/THEOREM_W3.md) 的精确配置（均基于 **Mode I Stop-Grad** 与 **`m5-c16-v1-le`** 输入契约）：

| 骨干模型 (HF Rev) | 宿主总参数 / `tie_emb` | 选定微调层带 (`FA` / `DN`) | Mode-I 累计参数（M1+W1/2/3） | W2 全对易版累计（M1+W1/2/3） | 推荐训练配置（Pattern-A `sub=8`） | 实测校准峰值显存 / 步耗 | 是否满足 M6 `65 GiB` 门槛 |
|---|---|---|---:|---:|---|---:|---|
| **`Qwen3.5-4B`** (`851bf6e8`) | `4,205,751,296` / `True` | `FA: 12,16,20,24,28` · `DN: 11,15,19,23` (`stop L10`) | **`1,486,878`** (`68.05%` Tier-P1) | `1,619,678` (`74.1%` Tier-P1) | `BF16` `kv_train` `mb=16` | **`40.028 GiB`** / `2,260.0 ms` *(Mode II: `51.855 GiB` / `2,818.8 ms`)* | **是** (`40.03 <= 65`) |
| **`Qwen3.5-9B`** (`c2022362`) | `8,953,803,264` / `False` | `FA: 12,16,20,24,28` · `DN: 11,15,19,23` (`stop L10`) | **`2,024,478`** (`92.65%` Tier-P1) | `2,095,838` (`<= 2.20M`) | `BF16` `kv_train` `mb=16` | **`53.426 GiB`** / `4,811.5 ms` *(Mode II: `67.270 GiB` 超 `65 GiB`!)* | **是（靠 Mode I 解锁）** |
| **`Qwen3.8-27B`** (`37561816`) | `26,895,998,464` / `False` | `FA: 20,24,28,32,36,40,48,56` · `DN: 19,23,27,31,35,39` (`stop L18`) | `3,377,630` (`0.0126%` host) | `3,692,830` (`0.0137%` host) | **`Mixed MLP-NF4` + `DN/FA BF16` + `Cayley FP32` ckpt `mb=16`** | **`45.267 GiB`** / `46,312.2 ms` *(纯 BF16 ckpt `mb=8`: `59.138 GiB`)* | **是（双模式均 `<= 65`）** |

---

## 五、诚实边界与 Phase II Modal A100 实测清单

为贯彻绝对严谨，明确区分**已闭环证毕项**与**必须留待 Phase II Modal GPU 实测项**：
1. **数学与代数结构（绝对成立，不依赖具体检查点）**：`C_4` 商群等变性、`RefSufficientPool` Schur 补充分统计量、复对称化零空间 $[P_{\perp, C}^{\mathbb{C}}, J_d]=0$ 与 Factor-2 Loewner 界、`GL(r)` 规范消除与 Lie 代数 $U(m)\cap \Pi_\perp$ 闭合、NF4 对 DeltaNet/Cayley 精度退化机理、`Option B1` stop-grad 梯度阻断规律、以及 `$12.88` 十进制费用包络。
2. **CPU 代理与真实 Stage 3D 数据混合实测（已在 [`verify_all.py`](verify_all.py) `10/10` 通过）**：`d=32/64` 混合骨干、`M1–M6` 物理分布/Pluto & CTTC 索引保护审计、`m5-c16-v1-le` 字节合同、`W1–W4` 相对构型分离度。
3. **尚待 Phase II A100-80GB 现场测定项（进入 [`STAGE4_PROTOCOL.md`](STAGE4_PROTOCOL.md) 闸门）**：
   - `Qwen3.5-4B / 9B / Qwen3.8-27B` 在真实多轮中/英文 + JSON 上的逐层 K-FAC 活化谱 $\{\sigma_j^{(\ell)}\}$ 与 `MixKL`；
   - 真实连接器在 `4B` 宿主上的实测候选 Jacobian $\sigma_3(M_{\mathrm{cand}})$ 及 Two-Probe Phase-A VoI 实测值 $\widehat{\Delta}_{\mathrm{host}}$（决定走 Branch 0 还是 Branch 1 `Surgical-96`）；
   - 真实 Pluto / CTTC 因果实采 IQ 数据噪声协方差矩阵 $\Sigma_{ij}$ 与端到端盲测指标、以及与 LEMMO Tier-P1/P2 的实车对照检验。

---

## 六、全量交付物与完整性锚点

- **总控执行协议**：[`STAGE4_PROTOCOL.md`](STAGE4_PROTOCOL.md)
- **顺序复现校验器与汇总结果**：[`verify_all.py`](verify_all.py) ｜ [`verify_all_summary.json`](verify_all_summary.json)
- **文件级 SHA-256 指纹清单**：[`MANIFEST.sha256`](MANIFEST.sha256)
- **关键工程实现与转录工具**：[`M3/ref_pool.py`](M3/ref_pool.py) ｜ [`transcript_utils.py`](transcript_utils.py)
