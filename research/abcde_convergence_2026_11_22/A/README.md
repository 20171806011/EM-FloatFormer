# EM-FloatFormer 目标观测的单次计分与 A/B 共用回归样本

完整证明及逐候选、逐比特读数见 [REPORT.md](REPORT.md)。本目录可独立加入 Git 仓库。脚本需要 Node.js 18 或更新版本，使用内置 JavaScript，无需安装包、访问网络或准备数据。所有浮点比特概率均为最终 LLR 温度校准前的值。

| 文件 | 用途 |
|---|---|
| [verify.cjs](verify.cjs) | 直接枚举离散 HMM 路径和整段载荷，逐比特核对两条后验公式，输出 JSON。 |
| [reference.json](reference.json) | 21 组观测、49,280 个全隐变量组合、156 个逐位真值。离散概率以最简分数保存；高斯积分等少数检查是浮点数。 |
| [ab_fixture.json](ab_fixture.json) | A/B 共用的 2×3 复 IQ、两候选信道、先验、转移、窗估方差、两个温度与 B 的逐分支预期。 |
| [verify_ab.cjs](verify_ab.cjs) | 独立枚举两行各自的全部 16-QAM 符号三元组和四条 HMM 路径，读取同目录的共用 JSON 并执行回归断言。 |

在此目录运行：

~~~sh
node verify.cjs
node verify_ab.cjs
~~~

第一条产生完整参考 JSON，并在结果不符合数学恒等式或冻结参考时抛错。第二条按分支比较局部密度、对数证据、比特联合密度和条件概率，再比较门权、有效权重、四位概率与硬判。预期输出 `status: PASS` 和 `numeric_checks: 196`。冻结和重拟合输入的标准见报告第 7 节。要接入 B，先读取 `input`，再与 `expected.local_frozen`、`expected.local_masked`、`expected.by_tau` 比较；勿向待测实现输入 `expected` 中的门权或解调结果。重新估方差的实现可将 `frozen_variance` 和 `masked_variance` 作为独立检查点。

理论基准提交 `f566f9b8b0ddfb468a4cb3fd64588bfa46c87212`。OFDM 读出及温度配置提交 `5f88a5be984493bcb6aa164d84253b6f29876eef`。本包只含合成算例与数学结果。
