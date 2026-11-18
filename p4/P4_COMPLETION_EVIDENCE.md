# P4 研究验收索引

截至 **2026-11-18 04:11 UTC**，正式二维和一维实验均已完成独立重放。原端到端套件在隔离副本通过 253/253 项，另一次隔离运行通过全部 100 项单元与对抗测试，包含 15 项新增白盒测试。S01 和 S03 的用户验收口径、连续轨迹结果、精度与硬件实测范围见下文。[原计划](../PROJECT.md) · [研究结论](STAGE3_RESULTS.md) · [联合重放](results/p4d/joint_replay.json)

## 逐项证据

| 需求与里程碑 | 功能及验证内容 | 可复查记录 |
|---|---|---|
| R1 · M1 | **F1** 四倍过采样 Rapp AM-AM/AM-PM；**F2** 时域多径、CP、多普勒 ICI；**F3** 零多普勒、线性 PA 与静态频域模型的相对误差小于 10⁻⁵。实现入口为 `rapp_pa`、`DynamicChannelSimulator`、`static_limit_parity_check`。 | [模拟器](simulator.py) · [M1 原 14 项测试](tests/test_p4_0_simulator.py) |
| R1 · M1 | **F4** Bussgang 增益、失真方差和 LMMSE；**F5** 16-QAM 逐比特 LLR 校准；**F6** 软判决重构和迭代 PA 抵消。测试覆盖线性极限、AWGN 校准及失真迭代。 | [经典接收机](classical_baselines.py) · [M1 单元测试](tests/test_p4_0_simulator.py) |
| R1 · M1 | **F7** 经典接收机延迟。用原程序在 CPU 四线程上重测六个方法，批量分别为 1 和 64，每项预热 25 次、计时 100 次。12 行 CSV 与 JSON 的中位数、P95 及其他字段逐项吻合。测量时同机正在执行二维训练，绝对时间含并发负载影响。 | [计时程序](benchmark_latency.py) · [四线程完整计时表](results/stage3_latency_cpu4/latency_cpu.csv) · [运行环境及时间戳](results/stage3_latency_cpu4/timing.json) · [冻结前的单线程测试留档](results/latency_cpu.csv) |
| R2 · M2 | **F8** 导频 RMS 与复数归一化；**F9** 无偏置复线性层、Hermitian 注意力、相对相位 RoPE 与径向激活；**F10** 双向导频/数据注意力；**F11** BCE、信道 NMSE 和 PA 前向残差联合训练；**F12** 复尺度下信道等变、LLR 不变和硬判翻转检查。P4-A 原验收 53/53，另有高动态范围及零导频梯度回归。 | [一维模型](em_floatformer.py) · [P4-A 单元测试](tests/test_p4_a_equivariance.py) · [复尺度压力测试](tests/test_gauge_stress_adversarial.py) · [75 条比较观测](results/p4a_comparative_benchmark.json) |
| R2 · M2 | **LLR 校准与软译码**。一次波形仿真中配对五种一维读出，八个拟合共 440 个结果单元、225,280 次码块判决。相同权重的二维开发检查点另在 192 个新时隙上直接检验原生 LDPC。完整重放复现一维逐帧标记和二维结果。 | [两类接收机的计数及限制](P4A_CODED_CALIBRATION.md) · [重放收据](results/p4a_coded_calibration/replay.json) |
| R3 · M3 | **F13** 二维时隙与稀疏 DMRS；**F14** 连续采样多普勒及盲 CFO 同步；**F15** 原生半码率 LDPC 编码与软译码；**F16** 2D-LMMSE 与 IDD-PA；**F17** 分布内外的精确编码误块/比特计数。M3 原 13 项单测全部通过，23 个基准格点上重放了 115 行既有方法计数。 | [二维模型与基线](slot_2d.py) · [M3 单测](tests/test_p4_b_2d_fec.py) · [原始观测](results/stage3_gate_observations.json) · [资格收据](results/stage3_qualification.json) · [逐格点重放](results/stage3_benchmark.json) |
| R4 · M4 | **F18** 正交 2³ 消融，匹配参数预算，另有固定宽度开发对照。正式十种子共 80 次拟合，每次在相同比特上测三个增益。独立重训复现全部权重指纹及 240 项整数计数。 | [八臂配置](p4cd_factorial_protocol.json) · [逐种子效应](results/p4d/factorial/summary.json) · [独立重训收据](results/p4d/factorial_replay.json) |
| R4 · M4 | **F19** FP32、BF16、FP16 真实投影矩阵乘法，归一化、FFT 及读出保留宽精度。100 个二维时隙中，BF16、FP16 相对 FP32 的翻转率分别为 0.03726%、0.00481%。固定 BF16 引擎跨七个尺度时翻转数和最大 LLR 差均为 0。全投影 BF16 在二维原样本最大 LLR 偏差为 0.20229，超过 0.15。仅在块内启用 BF16 后，两批二维最大误差均小于 0.012。一维原样本仍有 3/4,096 位翻转。主机无原生 BF16 矩阵加速，实测吞吐为 FP32 的 0.436 倍；支持此类加速的硬件能否达到蓝图的 1.2 倍仍待实测。 | [投影实现](stage3_precision.py) · [逐 SNR 结果及硬件能力](results/stage3_precision.json) · [一维机制与精度研究](P4C_1D_REPORT.md) · [真实 BF16 混合精度复核](results/mixed_precision_followup/reproduction.zip) |
| R4 · M4 | **F20** 正式协议在结果产生前冻结了 117 个文件；**F21** 以十个种子为配对单位，枚举 1,024 种符号翻转，一维三项与二维六项各自执行 Holm 校正；**F22** 按提交 `f878a736` 核对 230 个历史路径并复核旧清单。最终分别复现一维三项和二维六项。 | [冻结文件、环境与历史指纹](results/p4d/P4D_FREEZE.json) · [正式执行入口](confirmatory_protocol.py) · [一维正式结果](results/p4d/factorial/summary.json) |
| M5 · 跨里程碑 | **F23** 原四层共 253 项，冻结前 250 通过，0 断言失败、0 跳过，3 项因当时缺少正式冻结或正式结果报错。正式研究后，同一套件在隔离副本通过 253/253 项。**F24** 新增 15 项白盒测试实跑全部通过，覆盖完整性、零/混合导频梯度、载波失锁边界、真实推理的标签替换、低精度乘法及损坏证据拒绝。 | [原套件](../tests/test_p4_e2e.py) · [冻结前逐项记录](results/stage3_e2e.json) · [Tier 5 测试](tests/test_stage3_adversarial.py) · [全部单元及对抗测试收据](results/stage3_final_unit_tests.json) |

## M3 场景口径

- **S01 高速列车**按用户选定的独立衰落时隙验收，每个时隙内仍连续传播。四时隙交织、半码率长码在默认样本错 0/4 块，两种多普勒设置各在 12 组开发样本错 0/48 块。四时隙共用同一连续轨迹时，默认样本的同一映射均错 4/4 块。另做的全波形信息上界高于业务量，无法判定连续条件在物理上不可实现。[独立及连续波形](results/stage3_link_hst_summary.json) · [连续信道真值诊断](results/stage3_hst_feasibility.json)
- **S03 频偏**按用户选定的载波象限失锁验收。112 行均可观测且处于正确象限，104 对相邻行无象限变化，平均残余相位 0.936°。另记录到载荷 I/Q 符号位错 71/6,656。旧资格记录保留原符号位判据及当时的失败结果。[相位与符号位计数](results/stage3_cfo.json) · [旧资格记录](results/p4b_gate.json)
- **S02、S05、S06、S07、S11**的公开输入、整数分母和阈值存于资格记录。S05 的六步双向模型较断路后重训改善 0.78 dB，十二步纯物理模型又领先完整六步模型 0.77 dB。S06 在 156 块中错 25 块，关闭学习初值也错 25 块。S11 采用 50 时隙交织，在原样本错 4/1,600 块，另 12 条轨迹中有两条超过 1%。这些数据限定了对学习机制和跨信道稳定性的结论。[工作量记录](results/stage3_gate_observations.json) · [S05 对照](stage3_sparse_report.md) · [S06 对照](results/stage3_s06_hybrid.json) · [S11 复测](results/stage3_link_variability.json)

## 正式研究与终验

**一维机制研究已完成。** 对 80 份逐臂记录独立加总 240 组整数误码并复算三项双侧检验，结果与收据一致。导频归一化和 Hermitian 注意力分别使 BER 下降 2.615、1.086 个百分点，两项 Holm 校正 p 均为 0.00586。该一维短训练条件下，双向传播使 BER 上升 0.282 个百分点，校正 p 为 0.015625。一个物理模块在特定架构中的作用需要该架构自己的对照。[逐种子统计](results/p4d/factorial/summary.json) · [独立重放](results/p4d/factorial_replay.json)

**二维编码研究已完成。** 十个预设种子各训练 600 步，合计 6,000 次优化更新。三种信道条件、五个接收机共有 150 行逐种子整数记录，每个种子及条件评估 416 个码块、26,624 个信息位。独立计算核对了全部模型的权重指纹、六项检验的全部 1,024 种符号组合及其 Holm 校正。最终测试共通过 253 项，无失败、报错或跳过。[六项结果及逐条件计数](STAGE3_RESULTS.md) · [原始整数](results/p4d/raw_by_seed.csv) · [二维独立重计](results/p4d/coded_replay.json) · [联合重放](results/p4d/joint_replay.json) · [独立统计核验](results/stage3_final_audit.json) · [最终逐项记录](results/stage3_final_e2e.json)

<!-- P4-D FINAL c021f4ad4251a796125143f74405ae577bb0ff77a28e4dc805a49eff39c15036 -->
