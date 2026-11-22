# EM-FloatFormer 离散与连续 CFO 基线

**B 的集成入口**是 `native_bridge.receive(y, pilots, contexts, method, phase_only=False)`。两种物理头用 `method="discrete"` 或 `"continuous"`，学习头传入冻结的 CPU eval 模型。返回 float32 [1400,4]，按固定提交 DATA_COORDS_101 排列，正 LLR 表示比特 1。包可放在仓库的 `research/em-floatformer-c/`，独立 JS 原型无需安装依赖。

## 运行

从仓库根目录执行（Node ≥18）。

~~~sh
node research/em-floatformer-c/estimator.cjs
node research/em-floatformer-c/synthetic.cjs --check
PYTHONPATH=research/realdata_ofdm_stress_and_theory python3 research/em-floatformer-c/smoke_native.py
~~~

前两条独立生成合成网格，--check 自动与 checks.json 和 ablation.json 比对（浮点末位可有平台差异）。第三条需要可 import 固定提交的 Python 接收机及其依赖（NumPy、SciPy、CPU PyTorch 等）。smoke 以 JS 的确定性合成网格调用仓库真正的 differentiable_mixture_llrs_compact，比对 1400×4 个 LLR，检查两折掩码、方差刻度、预先权重和目标干预。已在 V8 执行两套 JS 校验、生成 native fixture；原生 Python/PyTorch smoke 当前**未运行**。不要把 JSON 的既有检查结果当作 native smoke 回执。

## 如何接真实网格

集成调用前将 research/em-floatformer-c 和 research/realdata_ofdm_stress_and_theory 都加入 PYTHONPATH。同一个 14×128 fftshift 网格和同一组可见导频只创建一次两折上下文。

~~~python
from native_bridge import build_contexts, receive
from fold_frontend import make_builder
build_one_fold = make_builder(frozen_train_ref_bank)
ctx = build_contexts(y, pilots, build_one_fold)
D = receive(y, pilots, ctx, "discrete")
C = receive(y, pilots, ctx, "continuous")
# 严格相位单项对照：两种学习头只提供相位，权重和方差也冻结到物理组的值
plain = receive(y, pilots, ctx, frozen_plain.eval(), phase_only=True)
transport = receive(y, pilots, ctx, frozen_transport.eval(), phase_only=True)
~~~

包内 make_builder 复用固定提交的信道、特征、HMM，并对噪声功率、NBI 窗和聚合使用支持掩码；仅接受匹配的源码 blob SHA。也可传自定义 build_one_fold(view, valid, pilots, fold)，返回 hLS/hCA [101] complex、nv [14,101] real、logw [14,17]，学习头另需 cplx [14,17,4]、scal [14,17,10]。logw 允许 −∞，每行在 17 候选上归一化、至少一项有限。历史参考库必须冻结。目标折按活动列序号奇偶划分，仅数据 RE 可作目标；view 已将本折目标和不可见导频置零。所有统计（含 3×3 窗、跨行特征、HMM 和功率上限）必须按 valid 排除无效值并按有效数归一化，不能把零当成观测。已证明目标前的外部上下文也可直接传 receive，调用方负责来源。使用原始全网格缓存会破坏该保证。

D/C 共用 hLS、hCA、nv、logw 和同一原生解调器；两组的 lognv 均为零，只有连续相位 δφ 不同。使用 phase_only=False 运行完整学习头时，学习头可以更改权重和方差，必须另记为“完整头”；只有 phase_only=True 或另行固定方差才能声称有效方差也匹配。基于 Node 子进程的桥接专用于正确性与集成；若计时需报告 IPC 开销，或对所有端实现等价常驻路径。

## 复现证据

旧检查（checks.json，单轨迹 MAP）和新增消融（ablation.json，17 路按支持折产生的 HMM 权重）采用不同融合规则和方差，绝对错比特数不可跨表相减。全部合成，未接触 OTA 数据。

| 新消融，同一 ε=0.042，方差固定为 0.03 | 离散错比特 | 连续错比特 |
|---|---:|---:|
| 理想公共相位 | 0/5600 | 0/5600 |
| 理想公共相位 + 同一份 16 dB 噪声 | 151/5600 | 124/5600 |
| 时域 CFO，含 ICI | 0/5600 | 0/5600 |
| 时域 CFO，含 ICI + 同一份 16 dB 噪声 | 173/5600 | 114/5600 |

上述每格一帧，因子固定、发送符号相同。噪声在输出频域相加；最后两格使用 IFFT→CP→时域旋转→去 CP→FFT。HMM 是仅针对合成自检的 17 路参考实现，按未留出 RE 求似然，p_stay=0.8、ll_temp=6（复用固定提交的 Dev 校准），合成历史分支等于 LS，无历史数据；用目标折任意替换重算了两折全部上下文和两个头，输出参数完全一致，目标 LLR 对目标替换有响应。新消融在含噪两格分别减少 27 和 59 个错误，不能估算 OTA 收益比例。

推导、模型范围、失败条件及正式精度/延迟记录协议见 REPORT.md。固定提交为 5f88a5be984493bcb6aa164d84253b6f29876eef，旧版搜索对照为 ba39dd5968dbaee7b7a226a9bc78d536278fdee8。此包无数据加载器、训练命令或推送操作。

B 的 audit.excluded_bundle 采用单目标 RE 重建。本包采用按活动列奇偶的两折重建，四组必须同折才能把差额归于方法；单 RE 结果只能作独立核对。被冻结的 Group B Dev 配置（τ=6, p_stay=0.8）见 [校准记录](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/results/group_b_results.json#L4-L14)。本合成包输出原始 LLR，不应用 B3 的单独输出温度。
