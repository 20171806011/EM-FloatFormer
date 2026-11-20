# P10 · 用未知载荷重建信道

在 9 dB、三条已知延迟、B 帧含一个导频块的合成链路上，两批独立种子共产生 4,096 个完全重置码块。P9 和旧版单径修复同时译错其中 2,666 块。让五个未知载荷块的复相关参与估计后，新分支救回 **1,604 块（60.17%）**。使用同一轨迹库、同一导频和同一译码器，仅靠导频救回 153 块。两行独立载荷用于码字验证切换，最终接收机在上述共同失败中救回 1,603 块。

完整接收机在完全重置、单径跳变、连续三种机制等权的 12,288 个码块上，错块数由 4,079 降到 2,547，BLER 由 33.19% 降到 20.73%。其中完全重置贡献了 1,504 个净纠正。载荷对信道估计的贡献由仅导频对照衡量。端到端结果还包含切换规则的贡献。

| 研究材料 | 内容 |
|---|---|
| [原始路径注意力方案](path_attention_design.md) | 研究问题、数学推导和拟议的架构对照 |
| [载荷复相关与单径修复](covariance_repair/README.md) | 复协方差评分、物理候选、原生 LDPC 与逐链路记录 |
| [重置门控研究](reset_gating/README.md) | 物理门控、学习对照、冻结权重、校准和独立复核 |
| [完全重置后的载荷辅助重建](payload_reconstruction/README.md) | 轨迹估计、匹配对照、主评估和独立重复、误差区间、控制实验、复现命令 |
| [重建结果图](payload_reconstruction/reconstruction_results.png) | 两批共同失败的救回情况和其他主要结果 |

## 从本分支核对记录

在仓库根目录启动 Bash，使用各报告列出的依赖环境。以下命令从两批逐块记录重新计算摘要，并和随附结果逐字节比较。全量波形生成和译码命令见各实验报告。

```bash
export PYTHONPATH="$PWD:$PWD/p10/payload_reconstruction/prior"
export OPENBLAS_NUM_THREADS=1
cd p10/payload_reconstruction
python test_estimator_math.py
out_dir=$(mktemp -d)
python audit_rebuild.py --input rebuild_raw.json --out "$out_dir/primary.json"
cmp rebuild_audit.json "$out_dir/primary.json"
python audit_replication.py --input rebuild_replication.json --out "$out_dir/repeat.json"
cmp rebuild_replication_audit.json "$out_dir/repeat.json"
```

P9 基线为 [faeea639](https://github.com/20171806011/EM-FloatFormer/tree/faeea639727aab72740cdd0a719106f83824226e/p9)。重置门控和重建目录分别保留原始 SHA-256 清单，重建所用的上轮冻结模型在其 `prior/` 子目录。两批复验共享同一合成器和先验。现场链路、其他信噪比和未知延迟需要单独验证。
