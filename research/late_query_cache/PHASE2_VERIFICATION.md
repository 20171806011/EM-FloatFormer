# Phase 2 · 核验记录

## 核验版本与命令

核验源码为 [6c0ba3fc1332](https://github.com/20171806011/EM-FloatFormer/tree/6c0ba3fc1332508551c9dc8091d3a6f4b597dfe8/research/late_query_cache)。该次 [GitHub Actions 的完整编译日志](https://github.com/20171806011/EM-FloatFormer/actions/runs/46108358841/job/140866095981) 的任务状态为 `success`。Lean 版本为 `4.24.0`，编译器提交为 `797c613eb9b6d4ec95db23e3e00af9ac6657f24b`。Mathlib 固定在 `f897ebcf72cd16f89ab4577d0c826cd14afaafc7`。最终文档归档沿用下列源码对象和同一编译方法。

在已安装 elan 的环境下，从本仓库的 Linux 或 Bash 检出运行下面的命令。最后一条脚本会在固定 Mathlib 工程中逐文件实际编译为 `.olean`，任何编译失败或公理输出含 `sorryAx` 均返回失败。复核旧提交时，先检出对应表中的提交，并运行该次提交所带的相同路径脚本；早期失败原因见文末。

```bash
git switch --detach 6c0ba3fc1332508551c9dc8091d3a6f4b597dfe8
elan toolchain install leanprover/lean4:v4.24.0
git clone --depth 1 --branch v4.24.0 https://github.com/leanprover-community/mathlib4.git .phase2-mathlib
bash research/late_query_cache/check_phase2.sh .phase2-mathlib
```

脚本在 Mathlib 根目录中按文件调用 `lake env lean --root="$stage" -o "$stage/${name%.lean}.olean" "$f"`。六个源码对象如下。

| 源码 | Git blob |
|---|---|
| [Phase2Coding.lean](Phase2Coding.lean) | `9779e04fbacd181b00af84bfd692a38e1c969fcf` |
| [Phase2Confusion.lean](Phase2Confusion.lean) | `abd5fff6133eafd84b3a6459366ffa648fee2663` |
| [Phase2Cycle.lean](Phase2Cycle.lean) | `7683a64a283b8d19786a2fd25da911b13ef54dba` |
| [Phase2Fibers.lean](Phase2Fibers.lean) | `d3834c40c591b2998ef3a4a1cb3ff53d7ed33acc` |
| [Phase2Guess.lean](Phase2Guess.lean) | `d6c89cd15509e1069d7d2bc45ada554a9bcc1f47` |
| [Phase2Risk.lean](Phase2Risk.lean) | `82353011b660886c967bb86dd953b02cb6eee5ac` |

## 已经形式化的内容

| 文件 | 导出的核验声明（省略命名空间） | 数学内容 |
|---|---|---|
| [Phase2Coding.lean](Phase2Coding.lean) | `cycleDecodeExact`, `recoverBinNear`, `guardedCycleDecodeExact` | 任意半字宽的五环异或解码、相邻误差下的模 3 粗格恢复、二者复合后逐字精确重建。 |
| [Phase2Confusion.lean](Phase2Confusion.lean) | `successIndependent`, `graphPosteriorEntropy` | 一般有向邻域下成功词集无混淆；任意实数后验的成功/失败两部分熵上界，以独立数和全集大小计。 |
| [Phase2Cycle.lean](Phase2Cycle.lean) | `cycle5SupermodularConverse`, `cycle5WithUnequalDeficits`, `graphRowSound`, `fractionalGraphDual` | 任意实值单调超模函数上的五环不等式、五个不同缺额的稳健版，以及任意有限有向邻域的线性对偶证书可靠性。未宣称 Shannon 型证书完备。 |
| [Phase2Guess.lean](Phase2Guess.lean) | `independentTableMarginal`, `logMassBound`, `independentStrategyMass`, `confusionEntropyConverse` | 分上下文的整张随机猜测表可实现指定边缘律，有限实数加权对数和不等式、独立选表的联合成功质量界和熵形式的逆定理。 |
| [Phase2Risk.lean](Phase2Risk.lean) | `floorBinWindow`, `normalSubgaussian`, `twoSidedTail`, `splitRisk`, `twoNeighborDoubleScale`, `calibratedRisk`, `finiteGaussianTails`, `finiteNeighborDoubleScale` | 取整稳定性、标准高斯边缘律的次高斯与双尾界、Bochner 积分的好坏事件风险分拆、两邻居与任意有限邻域的均方误差界及精度校准。尾概率并集界不需要相互独立。 |
| [Phase2Fibers.lean](Phase2Fibers.lean) | `projectHits`, `intersectionStrip`, `overlapVolume`, `pairIncidenceIntegral`, `squareMass`, `collisionSecondMoment`, `expMixedLower`, `expEndpointDiameter`, `measureFromPairwise` | 双线性坐标下的窄带和截面 Lebesgue 测度界，乘积测度的二阶矩碰撞界；对正文指数传感器的四点混合增量下界、任意两解码表的截面直径界和直径至外测度的转换。 |

### 成功运行的 `#print axioms` 完整输出

```text
'LateQueryPhase2.Coding.cycleDecodeExact' depends on axioms: [propext, Quot.sound]
'LateQueryPhase2.Coding.recoverBinNear' depends on axioms: [propext, Quot.sound]
'LateQueryPhase2.Coding.guardedCycleDecodeExact' depends on axioms: [propext, Quot.sound]
'Phase2Confusion.successIndependent' depends on axioms: [Quot.sound]
'Phase2Confusion.graphPosteriorEntropy' depends on axioms: [propext, Classical.choice, Quot.sound]
'LateQueryPhase2.cycle5SupermodularConverse' depends on axioms: [propext, Classical.choice, Quot.sound]
'LateQueryPhase2.cycle5WithUnequalDeficits' depends on axioms: [propext, Classical.choice, Quot.sound]
'LateQueryPhase2.graphRowSound' depends on axioms: [propext, Classical.choice, Quot.sound]
'LateQueryPhase2.fractionalGraphDual' depends on axioms: [propext, Classical.choice, Quot.sound]
'Phase2Fibers.projectHits' depends on axioms: [propext, Classical.choice, Quot.sound]
'Phase2Fibers.intersectionStrip' depends on axioms: [propext, Classical.choice, Quot.sound]
'Phase2Fibers.overlapVolume' depends on axioms: [propext, Classical.choice, Quot.sound]
'Phase2Fibers.pairIncidenceIntegral' depends on axioms: [propext, Classical.choice, Quot.sound]
'Phase2Fibers.squareMass' depends on axioms: [propext, Classical.choice, Quot.sound]
'Phase2Fibers.collisionSecondMoment' depends on axioms: [propext, Classical.choice, Quot.sound]
'Phase2Fibers.expMixedLower' depends on axioms: [propext, Classical.choice, Quot.sound]
'Phase2Fibers.expEndpointDiameter' depends on axioms: [propext, Classical.choice, Quot.sound]
'Phase2Fibers.measureFromPairwise' depends on axioms: [propext, Classical.choice, Quot.sound]
'Phase2Guess.independentTableMarginal' depends on axioms: [propext, Classical.choice, Quot.sound]
'Phase2Guess.logMassBound' depends on axioms: [propext, Classical.choice, Quot.sound]
'Phase2Guess.independentStrategyMass' depends on axioms: [propext, Classical.choice, Quot.sound]
'Phase2Guess.confusionEntropyConverse' depends on axioms: [propext, Classical.choice, Quot.sound]
'LateQueryPhase2.Risk.floorBinWindow' depends on axioms: [propext, Classical.choice, Quot.sound]
'LateQueryPhase2.Risk.normalSubgaussian' depends on axioms: [propext, Classical.choice, Quot.sound]
'LateQueryPhase2.Risk.twoSidedTail' depends on axioms: [propext, Classical.choice, Quot.sound]
'LateQueryPhase2.Risk.splitRisk' depends on axioms: [propext, Classical.choice, Quot.sound]
'LateQueryPhase2.Risk.twoNeighborDoubleScale' depends on axioms: [propext, Classical.choice, Quot.sound]
'LateQueryPhase2.Risk.calibratedRisk' depends on axioms: [propext, Classical.choice, Quot.sound]
'LateQueryPhase2.Risk.finiteGaussianTails' depends on axioms: [propext, Classical.choice, Quot.sound]
'LateQueryPhase2.Risk.finiteNeighborDoubleScale' depends on axioms: [propext, Classical.choice, Quot.sound]
```

以上是五环的抽象超模证书、概率论与几何内核等 30 条声明的公理依赖。编译还报告两类风格提示：一处后备 `ring` 不需要执行，以及两条定理自动带入的概率测度参数可进一步省略。它们均未阻止生成证明对象。

### 仍由书面证明处置的步骤

- 概率空间中条件后验实例化、完整的互信息/微分熵链、抖动熵与高斯最大熵、独立集平移覆盖的渐近速率、拼接系数和真实传感器截断的下界。
- 二进制有限编解码、Gaussian 尾估计与期望风险各自经过 Lean 核验；将整个实数数据通路接成一个端到端的最小广播率定理仍属于正文。一般矩阵的进位、实数系数平移修正及取极限也在此列。
- A2 的指数四角界与碰撞二阶矩已经核验；一般两动作坐标变换、近对角积分、泰勒可达码、Plancherel 和振荡积分去对数，以及对所有 $C^2$ 非仿射目标的斜率分类，由书面证明给出。没有把连续渐近定理列为一个已经过 Lean 的结论。

## CPU 交叉核对

两条 CPU 命令在附带本记录的归档提交运行。上方固定的 6c0ba3fc 提交用于复现 Lean 编译。

```bash
python research/late_query_cache/check_phase2_c5.py
python research/late_query_cache/check_phase2_fibers.py
```

五环检查枚举全部五个二位粗字、两个邻居各自的三个可修正误差，并包含一个强制超出保护范围导致错解的样本。后续高斯试验使用固定 seed 614029，每个噪声参数 8192 个连续输入。

```text
exhaustive neighbor decodes: 46080
eta fine_bits coarse_bits payload_bits corrected_crossings out_of_guard worst_query_empirical_mse
0 10 0 50 0 0 7.994366559134834e-08
0.5 10 2 55 7920 0 7.994366559134834e-08
1 10 6 45 4064 0 7.994366559134834e-08
```

曲纤维的构造码检查 17³ 个网格输入及五个查询，包括区间两端，半径 $r$ 和精度满足 $\varepsilon=r^2$。每个码长重复检查 24565 组。样本验证具体构造和实现，渐近下界来自证明。

```text
log2(1/r) fixed_bits source_question_pairs max_abs_error_over_r
4 8 24565 0.871305315597823
8 14 24565 0.8678788887218616
12 20 24565 0.8340382327405678
16 26 24565 0.8404407174821245
```

## 分提交核验历史

以下保留开发检查中的失败结果。提交对应的编译入口均为上面的 `bash research/late_query_cache/check_phase2.sh .phase2-mathlib`，CI 完整记录包含安装、命令及其退出状态。最后的文档/复现脚本归档不更改任何 Lean 源码。

| 提交 | 结果 | 日志 |
|---|---|---|
| [750f13a926](https://github.com/20171806011/EM-FloatFormer/commit/750f13a926e6465eab7fb572936fcba25edbddd8) | 工作脚本为 CRLF，进入 Lean 前退出，无公理输出。 | [运行](https://github.com/20171806011/EM-FloatFormer/actions/runs/46107804578/job/140864381223) |
| [8352dad09a](https://github.com/20171806011/EM-FloatFormer/commit/8352dad09a3ed8e6f27bf4409d021bab32c49f29) | 脚本仓库对象为 LF，检出规则仍将它转换为 CRLF，进入 Lean 前退出，无公理输出。 | [运行](https://github.com/20171806011/EM-FloatFormer/actions/runs/46107920614/job/140864718271) |
| [7ff00a7045](https://github.com/20171806011/EM-FloatFormer/commit/7ff00a704511f6e5ceeacb5700e20b90c62e7e86) | shell 检出修复。Lean 源码检出的 CR 进入了缓存依赖路径，编译前退出，无公理输出。 | [运行](https://github.com/20171806011/EM-FloatFormer/actions/runs/46107988104/job/140864934476) |
| [2aa9aab424](https://github.com/20171806011/EM-FloatFormer/commit/2aa9aab424f2df1316ddf8aa3aa934b5572c4218) | 已运行六文件的编译。一个集合成员目标及一个函数逐点相加目标需要显式展开，编译失败。该次输出不可用于认证。 | [运行](https://github.com/20171806011/EM-FloatFormer/actions/runs/46108161593/job/140865486639) |
| [6c0ba3fc13](https://github.com/20171806011/EM-FloatFormer/commit/6c0ba3fc1332508551c9dc8091d3a6f4b597dfe8) | 六份源码全部通过；上文逐条列出正式使用的全部输出。 | [运行](https://github.com/20171806011/EM-FloatFormer/actions/runs/46108358841/job/140866095981) |

<details>
<summary>中间失败提交 2aa9aab4 的全部公理输出（仅为错误复现，后由上面的成功运行取代）</summary>

```text
'LateQueryPhase2.Coding.cycleDecodeExact' depends on axioms: [propext, Quot.sound]
'LateQueryPhase2.Coding.recoverBinNear' depends on axioms: [propext, Quot.sound]
'LateQueryPhase2.Coding.guardedCycleDecodeExact' depends on axioms: [propext, Quot.sound]
'Phase2Confusion.successIndependent' depends on axioms: [Quot.sound]
'Phase2Confusion.graphPosteriorEntropy' depends on axioms: [propext, Classical.choice, Quot.sound]
'LateQueryPhase2.cycle5SupermodularConverse' depends on axioms: [propext, Classical.choice, Quot.sound]
'LateQueryPhase2.cycle5WithUnequalDeficits' depends on axioms: [propext, Classical.choice, Quot.sound]
'LateQueryPhase2.graphRowSound' depends on axioms: [propext, Classical.choice, Quot.sound]
'LateQueryPhase2.fractionalGraphDual' depends on axioms: [propext, Classical.choice, Quot.sound]
'Phase2Fibers.projectHits' depends on axioms: [propext, sorryAx, Classical.choice, Quot.sound]
'Phase2Fibers.intersectionStrip' depends on axioms: [propext, Classical.choice, Quot.sound]
'Phase2Fibers.overlapVolume' depends on axioms: [propext, Classical.choice, Quot.sound]
'Phase2Fibers.pairIncidenceIntegral' depends on axioms: [propext, Classical.choice, Quot.sound]
'Phase2Fibers.squareMass' depends on axioms: [propext, Classical.choice, Quot.sound]
'Phase2Fibers.collisionSecondMoment' depends on axioms: [propext, Classical.choice, Quot.sound]
'Phase2Fibers.expMixedLower' depends on axioms: [propext, Classical.choice, Quot.sound]
'Phase2Fibers.expEndpointDiameter' depends on axioms: [propext, Classical.choice, Quot.sound]
'Phase2Fibers.measureFromPairwise' depends on axioms: [propext, Classical.choice, Quot.sound]
'Phase2Guess.logMassBound' depends on axioms: [propext, Classical.choice, Quot.sound]
'Phase2Guess.independentStrategyMass' depends on axioms: [propext, Classical.choice, Quot.sound]
'Phase2Guess.confusionEntropyConverse' depends on axioms: [propext, Classical.choice, Quot.sound]
'LateQueryPhase2.Risk.floorBinWindow' depends on axioms: [propext, Classical.choice, Quot.sound]
'LateQueryPhase2.Risk.normalSubgaussian' depends on axioms: [propext, Classical.choice, Quot.sound]
'LateQueryPhase2.Risk.twoSidedTail' depends on axioms: [propext, Classical.choice, Quot.sound]
'LateQueryPhase2.Risk.splitRisk' depends on axioms: [propext, sorryAx, Classical.choice, Quot.sound]
'LateQueryPhase2.Risk.twoNeighborDoubleScale' depends on axioms: [propext, sorryAx, Classical.choice, Quot.sound]
'LateQueryPhase2.Risk.calibratedRisk' depends on axioms: [propext, Classical.choice, Quot.sound]
'LateQueryPhase2.Risk.finiteGaussianTails' depends on axioms: [propext, Classical.choice, Quot.sound]
'LateQueryPhase2.Risk.finiteNeighborDoubleScale' depends on axioms: [propext, sorryAx, Classical.choice, Quot.sound]
```

</details>
