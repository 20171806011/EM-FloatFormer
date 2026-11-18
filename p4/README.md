# EM-FloatFormer 第四阶段（P4）

第四阶段研究测试了带功放失真、频偏和时变多径的 OFDM 接收机。这里汇集 Antigravity 的原始设计、压力测试、历次门审和最后一次复核，以及 Codex 接续完成的接收机对照、正式十种子研究与独立审计。实验均为本地基带仿真。

**正式二维比较**在 10 dB、强功放失真下进行。对协议预设的高斯导频插值模型，六项配对比较均未观察到相对固定先验经典接收机的显著 BLER 优势。唯一达到 Holm 校正显著性的差异在同分布信道：EM 比两轮 IDD-PA 多错 55/4,160 个码块（+1.322 个百分点，校正后 p = 0.02344）。一维八臂实验另有三项机制检验。开发阶段的傅里叶及逐径相位接收机使用单独的数据和实验预算。[正式结果及图表](STAGE3_RESULTS.md) · [逐种子原始计数](results/p4d/raw_by_seed.csv)

| 阅读目的 | 材料 |
|---|---|
| 查看原计划、验收设置及修改原因 | [P4 计划](../PROJECT.md) · [验收证据索引](P4_COMPLETION_EVIDENCE.md) · [Antigravity 技术档案](antigravity/README.md) |
| 追踪一维前端、LLR 校准、注意力与数值精度 | [编码校准](P4A_CODED_CALIBRATION.md) · [一维机制报告](P4C_1D_REPORT.md) · [混合精度复现包](results/mixed_precision_followup/reproduction.zip) |
| 比较二维接收机、稀疏导频和信道模型 | [首轮与插值实验](P4_CONTINUATION_REPORT.md) · [接收机复验](P4B_RECEIVER_FOLLOWUP.md) · [稀疏导频对照](stage3_sparse_report.md) |
| 核对正式协议、权重与统计 | [预设协议](P4D_PROTOCOL.md) · [冻结清单](results/p4d/P4D_FREEZE.json) · [十个二维种子的权重及计数、一维八臂原始记录](results/p4d/) · [联合重放收据](results/p4d/joint_replay.json) |
| 区分各次复验 | [253 项端到端测试](results/stage3_final_e2e.json) · [100 项单元与对抗测试](results/stage3_final_unit_tests.json) · [Antigravity 最后一次复核的逐项状态](antigravity/post_mirror_verification.json) |

S01 采用四个独立衰落时隙验收，同时保留四槽连续轨迹的数据。S03 采用载波象限失锁验收，150 Hz、20 dB 的 112 个 OFDM 符号均保持锁定，载荷 I/Q 符号位仍错 71/6,656。较早的研究笔记保留撰写时的状态，最终结论见上方正式报告。

## 从公开仓库复算

使用完整 Git 克隆及 CPU 环境。冻结时记录的是 Python 3.12.7 和 PyTorch 2.12.1+cpu。依赖版本见 [requirements.txt](requirements.txt)。在仓库根目录执行以下命令。

```sh
python -m pip install -r p4/requirements.txt
python -B -m publication.verify_p4
python -B -m publication.verify_p4 --replay
```

第一条校验命令核对原 Git 历史、冻结依赖、十个最终模型的权重、两组 Holm 统计及既有测试收据。加上 `--replay` 后会从十个保存的二维模型重新生成接收信号并译码，同时独立重训一维十种子 × 八臂，逐一比对三种增益下的整数误码和模型权重。[公开版校验程序](../publication/verify_p4.py) · [原始独立审计程序](../publication/p4d_original_audit.py)

原冻结清单中的 117 项有 116 项保持原始字节。历史 ZIP 的公开副本保留 92 个成员，其中两份报告移除了内部会话引用，一份早期测试记录移除了开发机路径，其余 89 个成员原样保留。脱敏对照列出两个 ZIP 及每个成员的原始和公开 SHA-256。原协议入口严格要求原 ZIP 的指纹，因此从公开克隆复算请使用上面的公开入口。原始协议、权重、逐种子计数和正式统计均按原字节提供。[逐文件脱敏对照](revisions/public_redactions.json) · [历史 ZIP 公开副本](revisions/pre_stage3_p4.zip)
