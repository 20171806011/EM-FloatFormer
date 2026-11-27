# Stage 4 Route A：`N1+N2+N3+N4` 联合连接神经元微创手术（`CN-SFT`）`Qwen3.5-4B` 实车验证报告

> **验证运行 ID**：Modal A100-80GB [`ap-Zu6a3LiqZK3ygUlpqevRJY`](results/billing_post_cn_sft_4b.json)（`wall_s = 1132.59 s`）  
> **总评结论**：**`12/12` 硬性通过闸门全部通过（`hard_pass_gates.all_passed = true`）**  
> **宿主不可变指纹**：`Qwen/Qwen3.5-4B` · `host_sha256 = f39c9e1b7a5719c1397dcdce42d221215fb20edadb881d2f81e745673636f64d`（U1 训练与审计后比特级恒等）  
> **产出检查点**：`stage4_cn_sft_u1_s202611.pt`（`sha256 = e808279ee3afadfda7f0a8409462d372f630e5a5d01ce2048a5b9762efe9ee88`）  
> **基准数据源**：[`cn_sft_4b_probe_results.json`](results/cn_sft_4b_probe_results.json) · [`stage4_synthetic_blind_results.json`](results/stage4_synthetic_blind_results.json) · [`phaseB_surgical_vs_lemmo.json`](results/phaseB_surgical_vs_lemmo.json) · [`billing_post_cn_sft_4b.json`](results/billing_post_cn_sft_4b.json)

---

## 一、核心结论与 12/12 硬闸门实测清单

Route A 将四轨神经元微观手术理论（`N1` 编译态 Fermi–Dirac 21-神经元对偶 `SwiGLU` 电路、`N2` 64-prompt 双空间复零空间 K-FAC + 192 维 NoPE FA LoRA + `SpecCayley`、`N3` 非覆写 `ShadowGDNBallastBridge` 正交总线桥、`N4` 硬平台平稳 $SO(2)$-不变 Dead-Zone + Fisher Query Gate + 增广拉格朗日铰链）统一装配进非侵入式叠加适配器 [`cn_sft_adapter.py`](cn_sft_adapter.py)，并由 [`cn_sft_probe.py`](cn_sft_probe.py) 在真实冻结 `Qwen3.5-4B` 上完成全量盲测验证（`seed = 202611`）：

| # | 硬闸门 (`hard_pass_gates`) | 合同阈值 | 实测结果 (`U1 Full CN-SFT`) | 状态 |
|---|---|---|---|---|
| 1 | `pure_kl_zero` | `PureKL == 0.0`, `max_logit_diff == 0.0` | `pure_kl_max = 0.0`, `pure_max_logit_diff = 0.0` | **PASS** |
| 2 | `mix_kl_mean_le_1e3` | `mix_kl_mean <= 1.0e-3` | `0.00091643` (`9.16e-4`) | **PASS** |
| 3 | `mix_kl_max_le_3e3` | `mix_kl_max <= 3.0e-3` | `0.00188093` (`1.88e-3`) | **PASS** |
| 4 | `active_carrier_mix_kl_mean_le_1e3` | `active_carrier_mix_kl_mean <= 1.0e-3` | `0.00091643` (`9.16e-4`) | **PASS** |
| 5 | `active_carrier_mix_kl_max_le_3e3` | `active_carrier_mix_kl_max <= 3.0e-3` | `0.00188093` (`1.88e-3`) | **PASS** |
| 6 | `blind_g1_acc_ge_48` | `G1_acc >= 0.48` (`n = 50,304`) | **`0.64617` (`64.62%`, 与物理神谕 `diff = 0.0000`)** | **PASS** |
| 7 | `blind_g1_nll_le_1_15` | `G1_nll <= 1.15` | **`0.97885` nats** | **PASS** |
| 8 | `blind_g3_acc_ge_45` | `G3_acc >= 0.45` (`n = 1,024`) | **`0.62109` (`62.11%`)** | **PASS** |
| 9 | `q2_blackwell_ge_5pp` | `Q2_C4 fresh - valid >= +5.00 pp` | **`+7.62 pp`** (`0.74609 - 0.66992`) | **PASS** |
| 10 | `iut_vs_lemmo_p_lt_1e4` | IUT `p < 1e-4` (`131` 簇) | **`1.149e-92`** | **PASS** |
| 11 | `vram_le_65gib` | `<= 65 GiB` (`66,560 MiB`) | **`45,688.6 MiB` (`44.62 GiB`)** | **PASS** |
| 12 | `params_under_1_85m_and_tier_p1` | 可训练 `<= 1.85M` 且含 N1 稠密等效 `<= 2,184,964` | **`1,666,839` 可训练 / `2,089,239` 含 N1** | **PASS** |

---

## 二、全臂盲测对比汇总（`Qwen3.5-4B` 同源 `131` 簇盲测）

> 注：`Direct Physical Oracle`、`Zero`、`Naive-Additive`、`Equivariant-Host-Pure`、`Mode-I Surgical-96`、`LEMMO Tier-P1` 指标直接引自 [`stage4_synthetic_blind_results.json`](results/stage4_synthetic_blind_results.json) 与 [`phaseB_surgical_vs_lemmo.json`](results/phaseB_surgical_vs_lemmo.json)；`CN-SFT U0 / U1 / U2` 引自 [`cn_sft_4b_probe_results.json`](results/cn_sft_4b_probe_results.json)。所有主模型在 `G1 (n=50,304)`、`G2/G3/G4 (n=1,024)`、`Q2 C4 (n=512)` 与 `G1 Counterfactuals (n=1,024)` 上同源评估（`U2` 无 SwiGLU 对照按协议于 `G1` 评测 `32` 块 `n=12,288`）。

| 模型 / 实验臂 | `G1` (`n=50,304`) Acc / NLL | `G2` Acc / NLL | `G3` Acc / NLL | `G4` Acc / NLL | `Q2` Valid → Fresh (Blackwell 增益) | `Q2` Dup Dedup 差异 | `CF` Factual / Slot-Flip / Text-Flip (`symKL_slot`) | `PureKL` max / logit Δ | `MixKL` mean / max | 可训练参数 / 峰值显存 |
|---|---|---|---|---|---|---|---|---|---|---|
| **Direct Physical (Oracle)** | `64.62%` / `0.9411` | `46.78%` / `1.0222` | `62.11%` / `0.9796` | `47.75%` / `1.0159` | `66.60% → 74.41%` (`+7.81 pp`) | `0.0` | `64.36% / 64.36% / 64.36%` (`0.0`) | — | — | — |
| **`CN-SFT U0 (Zero-Shot)`** | **`64.62%` / `0.9716`** | `45.61%` / `1.0454` | **`62.11%` / `1.0046`** | `47.66%` / `1.0369` | `66.60% → 74.80%` (`+8.20 pp`) | **`0.0`** | `64.36% / 3.52% / 64.36%` (`1.4916`) | **`0.0` / `0.0`** | **`0.0` / `0.0`** *(Act: `0.0`)* | `0` 梯度步 |
| **`CN-SFT U1 (Full, 96 steps)`** | **`64.62%` / `0.9788`** | **`46.68%` / `1.0411`** | **`62.11%` / `1.0104`** | `45.80%` / `1.0314` | `66.99% → 74.61%` (**`+7.62 pp`**) | **`0.0`** | `64.36% / 3.52% / 64.36%` (**`1.5711`**) | **`0.0` / `0.0`** | **`9.16e-4` / `1.88e-3`** *(Act: `9.16e-4`)* | `1,666,839` (`2,089,239` 含 N1) / `44.62 GiB` |
| `CN-SFT U2 (No-SwiGLU, 64 steps)` | `25.51%` / `1.3490` *(n=12,288)* | `43.36%` / `1.0979` | `28.52%` / `1.4091` | `44.14%` / `1.0790` | `64.65% → 72.66%` (`+8.01 pp`) | `0.0` | `26.07% / 13.48% / 18.26%` (`0.2201`) | `0.0` / `0.0` | `9.95e-4` / `2.31e-3` *(Act: `9.95e-4`)* | Adapter `994,110` / `44.21 GiB` |
| `LEMMO Tier-P1` | `37.04%` / `1.2694` | `46.68%` / `1.0667` | `30.47%` / `1.3404` | `44.14%` / `1.0557` | `64.06% → 72.66%` (`+8.59 pp`) | `0.0` | `37.30% / 11.43% / 29.30%` (`0.3908`) | **`7.06e-4` / `0.5`** | `3.70e-3` / `8.58e-3` | `1,642,448` / `41.93 GiB` |
| `Mode-I Surgical-96` | `31.36%` / `1.3197` | `44.73%` / `1.0823` | `33.01%` / `1.3856` | `44.53%` / `1.0638` | `64.06% → 72.46%` (`+8.40 pp`) | `0.0` | `31.15% / 12.60% / 26.07%` (`0.3389`) | `0.0` / `0.0` | `4.19e-3` / `1.30e-2` | `1,725,937` / `73.48 GiB` |
| `Equivariant-Host-Pure` | `24.48%` / `1.3259` | `43.26%` / `1.1004` | `26.07%` / `1.3853` | `44.14%` / `1.0873` | `65.04% → 72.46%` (`+7.42 pp`) | `0.0` | `24.41% / 13.48% / 19.92%` (`0.1795`) | — | — | `488,400` / `50.22 GiB` |
| `Naive-Additive` | `19.39%` / `1.5252` | `41.11%` / `1.2795` | `19.04%` / `1.5075` | `43.07%` / `1.2620` | `29.49% → 33.40%` (`+3.91 pp`) | `0.0` | `19.14% / 16.02% / 18.65%` (`0.0110`) | — | — | `488,400` / `50.22 GiB` |
| `Zero` | `21.56%` / `1.4269` | `35.84%` / `1.3289` | `24.32%` / `1.4101` | `34.38%` / `1.3418` | `28.13% → 27.93%` (`-0.20 pp`) | `0.0` | `23.44% / 23.44% / 21.19%` (`0.0000`) | — | — | `488,400` / `50.22 GiB` |

---

## 三、`M2` 配对聚类统计与 Holm–Bonferroni 族系检验

基于 `131` 个独立盲测簇（`G1`, `n = 50,304`）与 `8` 个 `Q2_C4` 簇（`n = 512`），`U1 Full CN-SFT` 检验结论如下：

1. **`U1` vs `LEMMO Tier-P1`（G1 准确率与 NLL 双指标 IUT）**：
   - 准确率差：`+27.58 pp`（`0.27576`, 95% CI `[+26.63, +28.58] pp`, 单侧 `p = 1.149e-92`）
   - NLL 差：`-0.2905 nats`（95% CI `[-0.3000, -0.2809]`, 单侧 `p = 1.311e-97`）
   - **IUT 单侧 `p = 1.149e-92`** ｜ **Holm 调整后 `p = 2.299e-92`**（且 `LEMMO Tier-P1` 在通用文本上破坏了 `PureKL = 0`，`pure_max_logit_diff = 0.5`）。
2. **`U1` vs `Mode-I Surgical-96`（G1 双指标 IUT）**：
   - 准确率差：`+33.25 pp`（`0.33254`, 95% CI `[+32.23, +34.29] pp`, 单侧 `p = 2.051e-100`）
   - NLL 差：`-0.3409 nats`（95% CI `[-0.3509, -0.3307]`, 单侧 `p = 7.852e-103`）
   - **IUT 单侧 `p = 2.051e-100`** ｜ **Holm 调整后 `p = 6.153e-100`**。
3. **`U1` vs `Equivariant-Host-Pure`（G1 双指标 IUT）**：
   - 准确率差：`+40.14 pp`（`0.40142`, 95% CI `[+39.10, +41.20] pp`, 单侧 `p = 2.951e-107`）
   - NLL 差：`-0.3470 nats`（95% CI `[-0.3574, -0.3364]`, 单侧 `p = 2.984e-101`）
   - **IUT 单侧 `p = 2.984e-101`** ｜ **Holm 调整后 `p = 1.193e-100`**。
4. **`U1` vs `Direct Physical Oracle`（G1 簇配对）**：
   - `acc_diff = 0.0`（95% CI `[0.0, 0.0]`），NLL 差仅 `+0.03772 nats`（`0.97885` vs `0.94113`）。
5. **`U1` `Q2_C4` Blackwell 增援递增检验（`fresh_resample - valid`, 8 簇）**：
   - 准确率增益：`+7.62 pp`（`0.07617`, 95% CI `[+4.49, +10.74] pp`, 双侧 `p = 0.00368`, 单侧 `p = 0.00184`, Holm `p = 0.00184`）。

---

## 四、因果消融与核心结构诊断

### 1. Step-0 几何与机制实测诊断
- **RMSNorm $\gamma_f$ 补偿校准**：旧基线未补偿 $\gamma_f$ 时相对斜率误差为 `0.00819`；使用 `(M_cand D_{gamma_f})^+ F_real` 后 `b_align_fixed_err = 0.0`（`final_norm` 验证最大误差 `0.03099`）。
- **三路正交总线 (`e_bus`, `bus_dim = 22`)**：在 `blocked_dims = 515` 上实现候选方向、64 提示文本主子空间及附加阻断方向正交，实测泄漏上限 `cand_bleed_max = 1.72e-8`、`prose_bleed_max = 2.11e-8`、`extra_bleed_max = 3.04e-8`。
- **Layer 10 OOD 探针自适应分流**：Layer 9 (`stop_layer = 9`, 第 10 层) 线性脊回归探针在域外测试集上对 `phase_slots` $R^2_{\text{OOD}} = -1.11656$、全部特征 $R^2_{\text{OOD}} = 0.35877$，自动判定并触发 **Regime 2**（独立载波注入 + `ShadowGDN` 压载桥）。
- **Fisher LDA Query Gate**：在正例（EM 首轮末位，均值 `1.45594`，最小值 `1.40254`，`n_pos = 208`）与负例（纯散文、对话模板、混杂前缀全词元，最大值 `-0.16455`，混杂最大 `-0.22322`，`n_neg = 424`）之间严格线性可分（`linearly_separable = true`, `calib_pos_gate_min = 1.0`, `calib_neg_gate_max = 0.0`）。
- **模板后缀偏差单调安全校准（`U0`）**：对 `tid = 0..4` 仅在初始最大误差 `err0 > 0.10` 时修正（`tid=0: 0.40824 → 0.06449`；`tid=4: 0.14389 → 0.02833`），已低于 `0.10` 的模板（`tid=1,2,3`）免校准无过冲，`max_post_err = 0.07719`（`suffix_calib_converged = true`），尾层乘性增益比 $\rho_2 \in [0.9588, 1.0385]$。

### 2. 双对照因果消融（`U2` & `U3`）
- **机制对照 `U2`（移除编译态 `N1 SwiGLU` 电路与 `ShadowGDN`）**：保留相同 K-FAC 零空间 LoRA / SpecCayley、Fisher QGate 与增广拉格朗日控制，训练 64 步后，虽然 `PureKL = 0.0` 与 `MixKL = 9.95e-4` 依旧达标，但 `G1` 盲测准确率从 `64.62%` 崩落至 **`25.51%`**（`-39.10 pp`），`G3` 从 `62.11%` 跌至 **`28.52%`**，`G1` 槽位反事实对称 KL 从 `1.5711` 坍缩至 **`0.2201`**。这直接证明 `U1` 相对于 `LEMMO`/`Mode-I Surgical-96` 的 **`+27.58 pp ~ +33.25 pp`** 跨越来源于连接神经元物理电路与正交总线桥，而非 LoRA 微调偶然效应。
- **安全对照 `U3`（关闭 Fisher Query Gate 与平稳死区）**：在同一已训练 `U1` 权重上关闭门控开关，`PureKL` 仍因 `g_EM = 0` 短路保持 `0.0`，但混杂对话 `MixKL` 从 `9.16e-4` 恶化至 **`38.262` nats**（max **`65.187`**），Active Carrier MixKL 升至 **`0.07525`**（max **`0.4278`**），严格证明正交总线与微调叠加模块必须由双重结构门控包裹。

### 3. 实车调试中定位并修复的四项架构机制缺陷
1. **`_l0_pre` 零突变与纯净 `carrier_only`**：阻止 `_l0_pre` 修改返回给 `layers[0]` 的隐状态，使 Regime 2 `ShadowGDN` 扫描纯净 `carrier_only` 张量，从根源消除 `active_carrier_mix_kl` 残留。
2. **Decoder 残差保存顺序与 Mixer Post-Hook**：`Qwen3_5DecoderLayer.forward` 在进入 `post_attention_layernorm` 前即保存残差分支 `residual = hidden_states`；将总线注入迁移至 `layers[inject_layer]` Token Mixer（`linear_attn`/`self_attn`）后置钩子，并辅以尾三层门控 $P_{\perp,\mathrm{bus}}$ 清除，彻底解决总线注入被跳过及宿主各向同性噪声污染问题。
3. **因果化 `_gate_delta_by_qgate` 与 `require_dz` 隔离**：将 `qgate` 因果门控延伸至 `fa_cayleys`、`dn_cayleys`、`fa_loras[q_proj]`、`dn_gate_loras[in_proj_a]` 与 `slow_head_bias`，并令 `U1` 读出 `require_dz = True`。
4. **前缀 KV LoRA 差分学习率与增广拉格朗日铰链**：对写入 KV 的低秩旁路施加 `0.12×` 学习率与 `1.5×` L2 正则约束，在 $\epsilon = 5\times 10^{-5}, \rho = 3000, \mu_0 = 15$ 的增广拉格朗日铰链监督下将最终 `MixKL` 压至 `9.16e-4 <= 1e-3`。

---

## 五、Modal 算力结算封账与 Stage 5/6 平台迁移

- **Modal 本月最终结算（[`billing_post_cn_sft_4b.json`](results/billing_post_cn_sft_4b.json)）**：
  - 本次验证运行 (`ap-Zu6a3LiqZK3ygUlpqevRJY`)：`$0.80855863`（A100-80GB `$0.78611087` + CPU `$0.01562724` + RAM `$0.00682039`）
  - Stage 4 全阶段累计：`$8.78936278`（含 Stage 3D `$16.80382100` 后，本月总消耗 **`$25.59318378 / $30.00`**）
  - 剩余免费额度：**`$4.40681622`**（**实付账单 `Billed Cost = $0.00`**）
- **平台策略冻结**：遵循项目费用约束，**停止在 Modal 上启动任何新 GPU 作业**；全部 Stage 4 核心 `.pt`/`.npz` 产物仅通过 `$0` 只读 `modal volume get` 迁移至 GitHub Release，后续 Stage 5 & 6（`Real-SDR 4B`、`Qwen3.5-9B`、`Qwen3.8-27B Mixed-NF4` 与 `m5-stage6-blind-v1`）统一迁移至 **Google Colab** 执行。
