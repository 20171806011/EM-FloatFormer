# Stage 4–6 Modal A100-80GB 可执行工程协议 (`STAGE4_PROTOCOL.md`)

> **适用范围**：Phase II 实际云端 GPU 执行与盲测发布总规程，综合自 [`SYNTHESIS.md`](SYNTHESIS.md) 及 10 轨数学/系统定理（[`M1`](M1/THEOREM_M1.md)–[`M6`](M6/THEOREM_M6.md)、[`W1`](weight_mod/W1/THEOREM_W1.md)–[`W4`](weight_mod/W4/THEOREM_W4.md)），并已通过 [`verify_all.py`](verify_all.py)（[`verify_all_summary.json`](verify_all_summary.json)）离线前置核验。  
> **硬件与财务红线**：全部 GPU 作业统一运行于 Modal **`A100-80GB`**（`$0.000694/s` GPU 基准；含 2c/16GiB 系统核算费率 `$0.00075572/s`），严格锁定历史剩余免费额度 **`$12.88`**，要求 **`Billed Cost = $0.00`**，禁止任何超出 `$12.88` 包络线的任务调度。

---

## 0. 执行拓扑图（Stage 4–6 Strict DAG）

严格遵循 [`M6/THEOREM_M6.md`](M6/THEOREM_M6.md) 与 [`W4/THEOREM_W4.md`](weight_mod/W4/THEOREM_W4.md) 的有向无环依赖顺序执行；任一前置 Gate 失败均立即阻断下游 GPU 作业，实现零试错成本保护：

```text
[Gate 0: $0 CPU 预检 & 财务确认]
        │
        ▼
[4A: CPU 因果前端 + 双缓存 m5-c16-v1-le + RefSufficientPool] ($0.7677 CPU)
        │
        ▼
[4B: A100-80GB 显存/吞吐冒烟 & 实宿主 K-FAC / Jacobian 探针] (包含在 Pilot 首期)
        │
        ▼
[Gate 1: Phase-A 69 步 4-Arm 匹配先导实验 (固定 Seed 202611, 零 α 预筛)] ($0.8766 GPU)
        │
        ▼
[Gate 2: Two-Probe Phase-A VoI 决策分流]
   ├──(Probe A > 0.030 nats)────────────► [ABORT: 仅在 CPU/本地修复连接器, $0 后续 GPU]
   ├──(Probe A <= 0.030 & Δ_host <= 0.030)► [Branch 0: 冻结宿主完成 4C→4E/4F ($8.39–$9.50) + 全量 LEMMO Tier-P1 ($10.51–$11.37)]
   └──(Probe A <= 0.030 & Δ_host ∈ [0.04,0.40])► [Branch 1: Mode-I Surgical-96 ($10.01–$10.92) + 自适应 LEMMO L2/L3 ($12.56–$12.81 < $12.88)]
        │
        ▼
[Gate 3: 回归与统计终审 (PureKL=0.0, MixKL<=1e-3, host_sha256, M2 IUT+Holm)]
        │
        ▼
[Stage 5: Qwen3.5-9B & Qwen3.8-27B Mixed-NF4/BF16-DN/FP32-Cayley 多域与 Lie+Schur 合并]
        │
        ▼
[Stage 6: m5-stage6-blind-v1 主动 max_EIG 单次读取盲测]
```

---

## 1. Gate 0 — 启动前零开销前置闸门（`$0` GPU）

启动任何 Modal GPU 容器前，必须逐项记录并核对以下收据：
1. **Live 财务确认**：读取 Modal 实时账单确认当前 `Billed Cost = $0.00` 且可用免费额度足以覆盖所选分支上界（基准余量 `$12.88`）。
2. **宿主哈希与词表指纹锁定**：
   - `Qwen/Qwen3.5-4B`：`rev = 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`，`host_sha256 = f39c9e1b7a5719c1397dcdce42d221215fb20edadb881d2f81e745673636f64d`，`|V| = 248,320`，`tie_word_embeddings = True`，总参数 `4,205,751,296`。
   - `Qwen/Qwen3.5-9B`：`rev = c202236235762e1c871ad0ccb60c8ee5ba337b9a`，`|V| = 248,320`，`tie_word_embeddings = False`，总参数 `8,953,803,264`。
   - `Qwen/Qwen3.8-27B`：`rev = 37561816e66b5b9153b1b64c10968c18c1e6f4a0`，`|V| = 248,320`，`tie_word_embeddings = False`，总参数 `26,895,998,464`。
3. **4A CPU 因果与数据完整性校验（`0.7677 USD` 级别非 GPU 计算，受 [`M3`](M3/THEOREM_M3.md)/[`M4`](M4/THEOREM_M4.md)/[`M5`](M5/THEOREM_M5.md) 约束）**：
   - **Pluto (`QPSKlink.mat`)**：`train=3,000` (`[200,3200)`), `dev=1,000` (`[3500,4500)`), `test=2,240` (`[4800,7040)`)；边界 `100` 帧隔离；`16` 个独立测试簇 × `140` 帧；`stress_rows = 980`、`refs = [15, 18]`。
   - **CTTC (`Zenodo 4895133`)**：`train_target=1,440 / dev_target=864 / test_target=1,440`，参考各 `576`；`4,096` 循环保护；`72` 测试环簇；源比特足迹奇偶严格两两不相交；物理扩张后最小采集间隔 `>= 16,481` 样本；`f708_snr10`（`3,200` 原始行 + `480` 应力行）保留为独立防泛化检验集。
   - **双缓存包与参考池**：校验 [`M5`](M5/THEOREM_M5.md) `m5-c16-v1-le` 协议与[`ref_pool.py`](M3/ref_pool.py)（重复副本/干扰项/重排 `max_KL = max_TV = 0.0`，Schur 新息误差 `< 1e-12`）。

---

## 2. Gate 1 & 4B — A100 探针与 Phase-A 69 步 4-Arm 匹配先导实验

1. **4B A100-80GB 冒烟与实际宿主 K-FAC / Jacobian 探针**：
   - 采集实际中文/多模态样本的一趟前向 K-FAC 活化协方差 $C_{\mathrm{gen}}^{(\ell)}$（A100 上 `< 25 s`），构造复对称化零空间投影 $P_{\perp, C}^{\mathbb{C}} = \Phi(I_m - U_{\mathbb{C}, k_c} U_{\mathbb{C}, k_c}^*)$（4B 取 $k_c = 256$，保留 `1,024` 对完整复二重态）。
   - 提取候选连接器在真实 4B 宿主末层残差口处的 $M_{\mathrm{cand}}$ 奇异值 $(\sigma_1, \sigma_2, \sigma_3)$ 与 `W2` 对齐基 $B_{\mathrm{align}} = M_{\mathrm{cand}}^+ F_{\mathrm{real}}$（务必先完成 Phase-A 或零参数慢头稳态后再算 $B_{\mathrm{align}}$）。
2. **Gate 1 — Phase-A 69 步 4-Arm 匹配先导实验（固定 `seed = 202611`，预算 `$0.8766`）**：
   - 四个严格匹配训练臂（每臂每步同源样本 `12,288` 例，先运行 `69 / 384` 步 = 总步数 `276` 步）：
     1. `Zero`
     2. `Naive-Additive`
     3. `Equivariant-Host-Pure`
     4. `Equivariant-Physical+Residual`（并设 `Direct Physical` 作为解析物理对照）
   - 在完全独立的 `Dev` 验证集（`<= 4,000` 行）执行 `M6` 遗留无效性预筛：此步**不消耗任何盲测假设检验 α**，保证 [`M2`](M2/THEOREM_M2.md) 第 1 步 Holm 阈值下的条件Ⅰ类错误严格维持 `<= 1.667%`。

---

## 3. Gate 2 — Two-Probe Phase-A VoI 决策闸门与 Modal `$12.88` 预算锁定

根据 Phase-A 冻结宿主在独立 Dev 集上的表现，执行确定性二元/三元分流（依据 [`W4`](weight_mod/W4/results_w4.json) 与 [`W1`](weight_mod/W1/THEOREM_W1.md) Theorem W1.6）：

- **分支 A（连接器未过信息完备性）**：若线性/MLP 探针在 $E_{\theta_A}(x)$ 上相对真实物理后验的差距 **`Probe A > 0.030 nats`** $\implies$ **`ABORT_FIX_CPU_CONNECTOR`**（立即停止，禁止进入后续 384 步或权重微调，在本地 CPU 迭代连接器后再测）。
- **分支 B（冻结宿主已饱和 — 优先省钱全走 Branch 0）**：若 `Probe A <= 0.030 nats` 且冻结宿主路由缺口 **`Δ_host <= 0.030 nats`** $\implies$ **`STAY_FROZEN`**。
  - 执行 [`M6`](M6/THEOREM_M6.md) 锁定计划：**`Branch 0 Frozen Lock`**（续跑 4 个全连接臂至 `384` 步并完成 Pluto `53,760` 原始行 / CTTC `28,800` 原始行 / 应力 `8,240` / 合成盲测 `40,960` / 槽位与文本干预 `23,744`：无重试 **`$8.39064764`**，含 `900s` 重试 **`$9.07079564`**，硬顶 `$9.50`）+ 完整 `384` 步×`44,812` 行 Tier-P1 LEMMO（总计含重试 **`$11.36818444 <= $12.88`**）。
- **分支 C（确认存在宿主路由瓶颈 — 解锁微创 Phase B）**：若 `Probe A <= 0.030 nats` 且 **`Δ_host ∈ [0.040, 0.400] nats`** $\implies$ 触发 **`Branch 1 Mode-I Surgical-96`**。
  - 若 `Δ_host > 0.315 nats` 且实测 $\sigma_3(D_{\mathrm{answer}}) < 0.10$，先施加 **`W1+W3` 零参数双轨零空间慢头旁路**（`0` 新增参数，维持 Layer-10 `stop_grad`，将有效 VoI 覆盖上限拓宽至 `0.400 nats`）。
  - **Phase B 微创配置（4B Tier-P1 合规锁定）**：
    - 阶段模式：**Mode I / Option B1（Phase-A 连接器严格冻结 + Layer 10 截断梯度 `stop_grad`）**；仅当实测早期层 1–10 导致 $\sigma_3 < 0.05\sigma_1$ 才切换 Mode II（额外解冻 Layers `3, 7` 慢头门控，`40,976–82,944` 参数）。
    - 微调层带：`FA [12, 16, 20, 24, 28]` + `DN [11, 15, 19, 23]`；复对称化零空间 `P_{⊥,C}^C` + **FP32 Cayley $8\times 8$ Woodbury (`o_proj/out_proj`)** + **W2 Schur-Irrep 读出 (`C4=0.0, λ_orbit=0`)** + **`W1` 慢头温启动** + **`W3` 两层门控（`g_EM` 保 `PureKL=0`，`m_EM(t)` 仅作用于 K/V/β 写路径不屏蔽 Q 读出）**。
    - 精简核心累计 **`1,486,878`** 参数（或 `W2` 头内复 Cayley 全对易 **`1,619,678`**），严格位于 `2,184,964`（`2.20M`）LEMMO Tier-P1 对齐帽内；`4B` `BF16` Pattern-A `sub=8, mb=16` 峰值显存 **`40.028 GiB`**，步耗 **`2,260.0 ms`**。
    - 损失与步数：**`Bayes-KL + CF-Margin + 动态爬坡混杂锚定`** 跑 **`96` 步（`Surgical-96`）**，后接全量 `44,812` 行盲测与文本回归审计。
  - **严格十进制预算封顶（全部 `< $12.88`，`Billed Cost = $0.00`）**：
    - `Branch 1 Surgical-96 (Mode I)` 单独（含 `1,200s` 重试）：**`$10.91701960`**（余 `$1.96298040`）；
    - + **Tier-P1 LEMMO 对齐方案（二选一，优先 L2 留足重试安全垫）**：
      - **`Option L2（推荐锁定）`**：`384` 步 LEMMO × `20,480` 公共支持测试行 + `1,440s` 总重试池 = 无重试 `$11.47398524` / 含全重试 **`$12.56222204`**（余 `$0.31777796`）；
      - **`Option L3（极限全行）`**：`384` 步 LEMMO × 全量 `44,812` 行共享 `900s` 重试池 = 无重试 `$12.12617160` / 含重试 **`$12.80631960`**（余 `$0.07368040`）。

---

## 4. Gate 3 — 盲测统计门槛与零通用退化审计

每项声明必须同时通过 [`M2`](M2/THEOREM_M2.md) 与 [`W2`](weight_mod/W2/THEOREM_W2.md)/[`W3`](weight_mod/W3/THEOREM_W3.md) 的联合红线，方可视为成立：
1. **架构胜出与干预控制（固定检查点范围，IUT + Holm 跨 `{合成+Q2, Pluto, CTTC}` 族系控制）**：
   - `Equivariant-Host-Pure` 必须在配对比较中同时击败 `Zero` 与 `Naive-Additive`（准确率与 NLL 双指标 IUT），且通过固定文本下的签名交换/平衡洗牌干预检验；
   - `Equivariant-Physical+Residual` 必须击败 `Direct Physical` 与自身关闭残差的事前预测（禁止用事后盲测重校准作为证据）。
   - `Q2` 必须证明来自**不重复有效参考获取**的严格递增收益，且对**重复副本、独立干扰项、填充与输入列表重排**产生精确 `0.0` 状态变动。
   - **CTTC 推断单位与留出 guard**：在满足区块相关性检验时采用独立捕获（capture）作为聚类样本单元，并单独报告留出集 `f708_snr10`（作为跨频段零泄漏外推守护）。
2. **宿主不变量与文本无回退审计（Phase B 必查）**：
   - `host_sha256` 在磁盘与冻结层权重上保持比特级不变；
   - `g_EM = 0` 纯通用文本/代码/JSON 测试集 **`PureKL = 0.000e+00`**，`max_logit_diff = 0.0`；
   - `g_EM = 1` 携带 EM 前缀后的后继自然语言流利度审计满足 **`MixKL <= 1.0e-3 nats`**；
   - 若对比 LEMMO，严格区分 **Tier-P1（参数量 `<= 2.20M`、相同层带与 `m5-c16-v1-le` 数据同源对齐）** 与 **Tier-P2（LEMMO 原生未对齐 `20M–45M`）**，严禁混淆归因。

---

## 5. Stage 5 & Stage 6 — 大模型扩展与主动 EIG 单次读取盲测

1. **Stage 5（`Qwen3.5-9B` & `Qwen3.8-27B` 跨模型与跨捕获无重训合并）**：
   - **`Qwen3.5-9B`**：强制启用 **Mode I `stop_grad` L10**，使 `mb=16` 峰值显存锁定 **`53.426 GiB <= 65 GiB`**（禁止无截断 Mode II 在 `mb=16` 上运行以免冲至 `67.270 GiB`）。
   - **`Qwen3.8-27B`**：强制使用 **`Mixed MLP-NF4 + DeltaNet/FullAttn BF16 + Cayley FP32`** 混合精度（`45.267 GiB`），**严禁对 DeltaNet 门控或 Cayley $8\times 8$ Woodbury 核心实施 NF4 量化**（否则 MSE 恶化 `3,065.8×`、正交缺陷恶化 `20,162×`）。
   - **跨捕获融合（`Pluto + CTTC → 未见 f708_snr10`）**：禁止无约束 LoRA 因子直接平均，统一按 [`W2` 定理 8](weight_mod/W2/THEOREM_W2.md) 提取规范基 $( \widetilde{\mathcal{A}}^{(k)}, S_{\mathbb{C}}^{(k)} )$ 执行 **Canonical Lie+Schur 凸均值 Soup**（或 canonical TIES），严禁未经归一化对高对齐适配器使用过大权重算术相加。
2. **Stage 6（`m5-stage6-blind-v1` 主动参考选择盲测）**：
   - 统一在 `Qwen3.5-4B / 9B / Qwen3.8-27B` 与 `LEMMO_matched_adapter` 四宿主上，按 [`M5`](M5/THEOREM_M5.md) 严格协议：在单次读取预算 `read_budget = 1` 下，**读取前预先提交**候选窗口编号、`prior[2]` 及 `conditional_weights[2,16]`，仅读取所选 1 个 16-bit 窗口后输出后验 `posterior[2]`。
   - **终局胜出标准**：同一单次读取预算下，`max_EIG` 必须借助 [`M2`](M2/THEOREM_M2.md)/[`M5`](M5/THEOREM_M5.md) 配对 Bonferroni 检验证明显著优于 `random`、`nearest` 与 `fixed_best_on_dev`，且单独对比 16-bit (`m5-c16-v1-le`) 与 4-bit 历史缓存。
