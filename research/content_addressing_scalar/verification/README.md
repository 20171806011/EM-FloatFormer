# 复核标量内容检索的风险公式

从本目录运行以下命令。程序仅依赖 Python 3 与 NumPy。附带结果使用 Python 3.12.14、NumPy 2.3.5。

```sh
python verify.py --out reproduced
diff -u results/results.csv reproduced/results.csv
diff -u results/checks.json reproduced/checks.json
```

程序先计算结果，然后检查事前指定的等式。检查失败时退出码为 1。不同 Python、NumPy 或数学库版本可能产生末位差别，版本号会写入 checks.json。本次冻结的 PLAN.json 的 SHA-256 为 `35e983fdd3eec47812550d876c99e404dc14f71e18fab955eeb0a5b1041d1cd1`。九种场景和一个反例各用 32,768 个独立帧。场景内各估计器共用完全相同的帧。全部网格和判定规则在第一次模拟前写入 PLAN.json，所有策略的结果均保存在 results/results.csv。

## 如何读结果

- `att_mc,local_mc,global_mc,hard_mc,adaptive_mc` 是实际输出与目标的平均平方距离。固定窗口是前 k 个值，其 Bayes 估计器可读全部键。全局 Bayes 按各候选的精确二元信道似然加权。经典硬检索选择汉明距离最小的候选，平局取最小索引。自适应局部估计器先读全部键，再读后验概率最大的 k 个值，保留原后验权重。
- `att_formula_mc` 把每帧的权重和真索引代入风险等式右侧，再取平均。`local_formula_mc,global_formula_mc` 类似，使用给定键和查询后的条件风险。它们仍含侧信息抽样误差。`formula_residual` 与 `formula_residual_se` 是注意力实际损失减公式的逐帧差的均值与一个标准误。`att_minus_local_mc,att_minus_local_se` 同样采用逐帧配对。标准误是样本标准差除以帧数的平方根。
- `att_exact,local_exact,global_exact` 仅在完整枚举最多需要一万个距离元组时报告。借助候选的交换对称性，枚举时令真索引为首项，依各距离元组的联合概率积分；固定窗口风险使用 k/M 倍的全局后验平方和。空格表示未作完整枚举。`oracle_local` 是额外透露 J 所得的风险下界。
- 对每个场景，`margin_exact` 用真键与查询的二项距离和每个无关键距查询的独立公平二项距离直接求和。`margin_mc` 统计满足“至少相差 gap_bits 个汉明位”的比例。反向打分的策略取反方向的距离尾部。`bound` 代入精确尾部概率和温度得到，`certified` 表示它严格小于 `oracle_local`。在盲匹配的校准策略下，beta=0 表示均匀权重；CSV 仍报告原始相关分数的 margin 和 beta 向 0 的连续极限界。均匀估计器也可直接以全零 logits 在有限温度实现，此时没有正的打分 margin。
- `checks.json` 单列权重读取参考值时的反例。二元键每帧随机排列，查询是有效键的副本，故接收器可由观测准确匹配。读取匹配值后，一旦其功率超过全体均值的 t 倍就故意选择另一项。这里 t=1。若 D=v₀²/(v₀+σ²)，真实风险是 v₁+D(t+2)exp(−t)，把权重误当作条件独立所得风险是 v₁+2Dexp(−t)。差额源于越阈值时真值的条件均方功率增加。

每一帧实际构造二进制键和查询，再计算汉明距离。真实索引用于合成数据、检查 margin、核对风险恒等式。所有估计器只接收可用观测计算输出。有约束的估计器在读值时只读获准的坐标。运行时间包含数据生成和所有比较，不能用作不同接收器实现的延迟基准。
