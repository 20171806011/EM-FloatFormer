# 分别交给七个 Work 对话的提示词

每行是一份独立的中文任务书，含仓库与固定提交、研究问题、运行命令和独立分支的交付约定。准备新对话时，只需复制对应文件的全部正文。

| 组 | 可直接复制的提示词 | 已有独立交叉复核 |
|---|---|---|
| A | [迟到的导频能提供多少新信息](A.md) | [复核记录](../crossreviews/A.md) |
| B | [哪些相位与时间信息足以确定预测](B.md) | [复核记录](../crossreviews/B.md) |
| C | [硬标签与后验软标签应怎样混合](C.md) | [复核记录](../crossreviews/C.md) |
| D | [固定的 Q1/Q3 查找表是否选对每一类](D.md) | [复核记录](../crossreviews/D.md) |
| E | [一次正式宿主实验该测多少条独立波形](E.md) | [复核记录](../crossreviews/E.md) |
| F | [怎样判断宿主真的使用了频偏符号](F.md) | [复核记录](../crossreviews/F.md) |
| G | [一次模型读出能保证哪一种旋转对称性](G.md) | [复核记录](../crossreviews/G.md) |

## 已执行到哪一步

首轮 A–F 由六个内部研究任务分别完成，G 由主对话完成。七项 CPU 结果及 A 组导出的读出核均有 [GitHub Actions 复现记录](https://github.com/20171806011/EM-FloatFormer/actions/runs/45818584457)。第二轮由七个内部复核任务分题完成，每题均更换复核者。随后，项目内七个独立 Work 对话分别提交了推导、可执行核验和机器可读结果。下表固定各自的原始交付版本。完整 Work 输出已汇入同一研究分支。

交叉复核更新了 E 组的预算。在原先设定的效应、组内相关、方差及正态区组均值假设下，按正式分析的 t 区间规划，建议 131 个独立区组（8,384 条母波形）。129 组是原正态近似的答案。[假设、非中心 t 复算代码与敏感性分析](../crossreviews/E.md)

## 七个独立 Work 对话的交付

| 组 | 汇入的报告 | 原始分支提交 |
|---|---|---|
| A | [迟到的导频能提供多少新信息](../work_results/A/README.md) | [b13a1bec](https://github.com/20171806011/EM-FloatFormer/commit/b13a1bec3eb1e490bb24e5f7f1349cc3185499c1) |
| B | [哪些相位与时间信息足以确定预测](../work_results/B/REPORT.md) | [59b891e4](https://github.com/20171806011/EM-FloatFormer/commit/59b891e4efff66a8b2118e2a8726bba63543cb7f) |
| C | [硬标签与后验软标签应怎样混合](../work_results/C/REPORT.md) | [f0dfc725](https://github.com/20171806011/EM-FloatFormer/commit/f0dfc7252d9a6af8cd8f425e3b48ae639f11ad21) |
| D | [固定的 Q1/Q3 查找表是否选对每一类](../work_results/D/REPORT.md) | [e16ae577](https://github.com/20171806011/EM-FloatFormer/commit/e16ae57774a5a9d2ca67c33d00801da1bf3c1404) |
| E | [一次正式宿主实验该测多少条独立波形](../work_results/E/REPORT.md) | [aeb806df](https://github.com/20171806011/EM-FloatFormer/commit/aeb806df43190fec02a8c1cc5e91e87f8b03a171) |
| F | [怎样判断宿主真的使用了频偏符号](../work_results/F/README.md) | [67a7de9c](https://github.com/20171806011/EM-FloatFormer/commit/67a7de9c32b83f97c69cad2f860844fd20cdad5a) |
| G | [一次模型读出能保证哪一种旋转对称性](../work_results/G/README.md) | [54053854](https://github.com/20171806011/EM-FloatFormer/commit/540538546030b8bfbf3d8885b09adfb88fcacfaa) |

从仓库根目录运行 `python -B research/stage3c/verify.py --repo . --work` 复算七份新交付。数字来自推导、CPU 模拟或样本量设计。GPU 宿主的候选分数及正式盲测是下一阶段的实测项目。
