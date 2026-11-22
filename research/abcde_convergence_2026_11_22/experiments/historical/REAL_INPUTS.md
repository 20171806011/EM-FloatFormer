# 同一 435 MHz 录制的历史配对复算与实测回放边界

## 已运行的历史结果复算

直接读取 OFDM 固定提交 `5f88a5be984493bcb6aa164d84253b6f29876eef` 的完整 Group B 逐 TTI CSV（7,200 行）与 Group C 逐 TTI CSV（10,800 行），并与两份原 JSON 的全部五种工况、九种方法的 BER、原始及校准 NLL 独立对表（绝对容差 1e−10）。每种工况每个方法/种子为同一组 TTI 502…741、240 帧，每帧 1,400 个载荷 RE × 4 比特，共 1,344,000 个不同的比特位置。每个工况单报。三个训练种子的均值不会把独立实测 bit 或录制数量变成三倍。C1…C4 是同一次原始 IQ 录制的受控衍生。另一 435 MHz OFDM 分支也使用同一个数据文件 SHA-256，且划分方法不同；两套结果不能合并为独立实测验证。[原始历史 B CSV](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/results/group_b_per_tti.csv) · [历史 C CSV](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/results/group_c_per_tti.csv) · [旧 OFDM 分支同一数据摘要和划分](https://github.com/20171806011/EM-FloatFormer/blob/ba39dd5968dbaee7b7a226a9bc78d536278fdee8/research/realdata_sdr_ofdm/PROTOCOL.md#L14-L33)

以下对比原 B3 → 已归档的 Role A。Role A 的种子为 20261121、20261122、20261123，保留 TTI 655…665 同步异常。表内 BER 用 %，差用百分点，正表示 Role A 退化。NLL 单位为 nat/bit，按各臂原 Dev 温度；原始与校准两列分开。联合键为 (condition,TTI)，学习臂另含 seed。每 TTI 先对三个相同位置的种子差取均值，再对 12 个连续 20 帧块汇总，区间使用 df=11 的 t，双侧 sign flip 枚举 4096 种符号。p 值为描述性、未多重校正；块的独立/可交换性是假设。这里配对的是**归档的系统**；其中注意力、连续相位、重估、门控等变量并未单独固定，效应不能归给单个变量。

| 工况 | BER，B3 → Role A (%) | ΔBER 与块 95% CI (pp) | 原始 NLL，B3 → A | 校准 NLL，B3 → A | 块退/平/好 | 双侧 p |
|---|---|---|---|---|---|---|
| C0_raw_ota | 9.1534 → 10.6077 | 1.4543 [0.7997, 2.1089] | 0.20296 → 0.24739 | 0.20757 → 0.24782 | 12/0/0 | 0.000488 |
| C1_awgn_stress | 25.2702 → 24.8107 | -0.4594 [-0.6437, -0.2752] | 0.48362 → 0.47018 | 0.50740 → 0.47027 | 0/0/12 | 0.000488 |
| C2_cfo_doppler | 26.7733 → 14.4038 | -12.3695 [-14.1256, -10.6135] | 0.62200 → 0.29658 | 0.50670 → 0.29662 | 0/0/12 | 0.000488 |
| C3_phase_jump_jitter | 34.7958 → 33.9571 | -0.8387 [-2.2166, 0.5393] | 1.10921 → 0.54714 | 0.70755 → 0.54628 | 4/0/8 | 0.210449 |
| C4_nbi_sparse7 | 10.7475 → 10.3686 | -0.3789 [-0.5714, -0.1864] | 0.22944 → 0.21134 | 0.23977 → 0.21117 | 1/0/11 | 0.000977 |

## 救回和新增错的可识别范围

对某一帧和种子的 N=5600 位，设旧、新错误数为 a、b；重叠错误数 I 满足 max(0,a+b−N) ≤ I ≤ min(a,b)。救回=a−I、新增=b−I，净新增=b−a 精确。按每帧先约束再求和比分别用整组 BER 推断更紧。这是仅用错误边际的可行区间，不是置信区间或逐位观测。CSV 没有逐位真值与预测配对；相同 BER 甚至可伴随大量相互抵消的翻转。逐帧 BER 则足以精确统计帧级退/平/好。JSON 给出全部种子、五种成对系统切换、每种工况的净差和区间。下表取固定种子 20261121；其余种子在 JSON 中单列，不能将三次预测拼为独立 bit。

| 工况 / B3 → Role A | 每臂 bit | 净新增错 | 救回可行数 | 新增错可行数 | 帧退/平/好 |
|---|---:|---:|---:|---:|---:|
| C0_raw_ota | 1,344,000 | 14069 | 2334…123008 | 16403…137077 | 89/54/97 |
| C2_cfo_doppler | 1,344,000 | -166100 | 167846…359537 | 1746…193437 | 24/0/216 |
| C3_phase_jump_jitter | 1,344,000 | -12811 | 55263…457366 | 42452…444555 | 107/0/133 |

## 耗时

原 B 归档的所有六种算法合计 ms/TTI（共享前端）为 C0 37.21、C1 37.47、C2 39.01、C3 41.19、C4 53.02，整程序 393.71 s。C 归档测的是 Dev 上单 CPU 线程、batch=1、预热 5 次后的 200 帧神经头，跨种子平均 Role A 2.738、实 Transformer 0.861、MLP 0.412 ms/TTI。该计时不含缓存前端和最终解调；C 前端缓存总计 112.16 s，C 整程序含训练计 13471.91 s。不能用其相除推算任一单臂端到端加速比。`historical_pairs.cjs` 只复算历史文本，V8 本次计时约 0.39 s（两表已读入内存，无网络、无接收机推理）。[B 历史记录](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/results/group_b_results.json) · [C 计时代码](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/group_c_transport_attention.py#L832-L853)

## 来源、输入授权和可用性

公开数据源为 `rikluost/sdr_ofdm_dataset_v1@6f9821d6390d212884fea3d6621267ebe70c7246`；上游 README 的用途为接收机验证，给出公开加载示例。该提交完整树中未附 LICENSE，GitHub 仓库元数据 license 为 null；公开的验证入口和原始采集的再分发授权分别记录，不把“公开可读”当作开放再许可。本包不含 IQ、发送标签或权重，只含聚合结果、研究代码和摘要。交由使用者按其权限提供本地数据副本。[上游说明](https://github.com/rikluost/sdr_ofdm_dataset_v1/blob/6f9821d6390d212884fea3d6621267ebe70c7246/README.md) · [上游固定目录](https://github.com/rikluost/sdr_ofdm_dataset_v1/tree/6f9821d6390d212884fea3d6621267ebe70c7246)

上游对象 `ofdm_valset_4_8_sdr.pth` 为 27,684,430 字节，git blob `9b1aee8673ec48f42f52dc365fb01cc8b3e2b741`。两条历史研究路径都记录 SHA-256 `e760079a888f37d8c33b21f327646930b4f2cb4511e8a309bcde22c6b1332058`。本轮只独立核实源树对象、大小与两份历史摘要；没有把归档摘要误作刚刚重算的 IQ 哈希。此次可用的内容端点对该大对象返回空 body，文本 blob 端点无法解码其二进制，且没有可用的本地执行器。未取得 IQ 字节，因此**本轮实测输入 smoke 和 240 帧接收机回放均未执行**。当前固定目录没有 Group C 对应的模型文件；训练代码在内存复制、恢复最佳 state_dict，持久化的是指标和 CSV。其他研究目录的 .pt 不是这里角色/种子的部署权重。旧权重缺失仅阻塞对应学习臂，不阻塞拥有 IQ 和依赖时的物理 B3 回放。[固定数据与网格](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/ofdm_data.py#L1-L15) · [Group C 恢复](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/group_c_transport_attention.py#L784-L823) · [保存输出](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/group_c_transport_attention.py#L1089-L1107)

## 冻结的重跑接口

1. 历史文本复算只需 Node 18+ 和固定提交的 results 目录：`node historical_pairs.cjs /path/to/results > historical_pairs.json`。程序检查两张表每组行数、唯一键、分母可恢复的整数误码数、三角色三种子以及和原 JSON 的一致性。输入的四个 SHA-256 固定在脚本中。B/C CSV 在 git 中为 LF；对 LF 字节实算的 SHA-256 分别为 `eb2c29f3f0407ab5e8ff2b297ca80a004da4cff30fa9117cd8971f3d58442c7c`、`4fc0bf43b5d8043add359021c3dc9c2de125b790c1459e9767f275306ee00a17`。归档 MANIFEST 的两条 CSV 哈希对应内容一致的 CRLF 表示，两种表示已实算核对；两个 JSON 不需换行转换。应读取 git 原始字节以复现脚本输入。使用相同源码在 V8 中重走命令行入口的参数解析、哈希和聚合（文件接口由内存提供），与直接调用结果完全一致；复制一条记录替换另一条的负例被重复键检查拒绝。
2. 真实输入入口 `real_smoke.py` 依赖 Python 3.10+、CPU PyTorch、NumPy、SciPy、B 的 `audit.py` 和指定提交源码。调用前在受信目录准备数据，脚本在反序列化之前校验字节数和 SHA-256；沿用上游自然二进制映射、导频、参考库及 HMM。Train=0..339，Guard=340..359，Dev=360..481，Guard=482..501，历史 Test=502..741；参考库为 16 个、只由 Train 建立。统一合成修正验证通过后先分析 Dev t=360，三处事先固定的 (行,活跃索引)=(2,39),(7,45),(13,100)，每工况 12 个配对 bit。原格式观测按 C0/C2/C3、原 condition_seed 和相同信噪参数生成一次。先判决再以真值统计；脚本只输出计数与时间。三种读出沿用 (τ,stay)=(6,0.8)，以目标首次使用的精确模型对照另设 τ=1。少量 bit 用于验证工程路径，不能估计总体 BER。
3. 命令 `python real_smoke.py --source <fixed-ofdm-dir> --audit-dir <B-dir> --pth <authorized-local-file>` 只执行 Dev smoke。待会合、冻结、依赖与数据就绪后，显式 `--b240` 可选择以未改动的 evaluate_group_b_on_tti 重跑归档 B 基准，全 5 工况、每工况相同 240 帧，使用预存的 τ=6、stay=0.8、B3 LLR 温度 2.3154744812056767，以逐帧历史 BER 做回归。没有 Test 调参或新训练。该可选入口解决旧物理基准的复现，不生成缺失的 C 预测，也不自动执行严格留目标的全量消融。三种逐目标方案需要逐个目标构造掩码；固定真实 IQ 与模型前不得声称全量结果。

## 解释边界

C0 的所有块在该系统比较中均退化，C2 的所有块均改善；C3 的 BER 差区间跨零，校准 NLL 却可改善。这个模式与设置单独 C0 验收、把相位跳变与线性 CFO 隔离相容。要把 C2 改善归因给连续 CFO，要在相同目标排除规则、门控、方差、导频、候选、位读出及统计单位下运行单变量对照。历史比较只定位这个问题，不替代新对照。
