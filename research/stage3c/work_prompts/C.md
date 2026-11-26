# Work 任务 C · 硬标签与后验软标签应怎样混合

你负责 EM-FloatFormer 的 C 题。请独立完成推导、执行数值核验，并将结果提交 GitHub。其他六组可以同时开展。

**起点**　在 [EM-FloatFormer 的冻结提交](https://github.com/20171806011/EM-FloatFormer/tree/bbc76e3813a24840b9821c59cb8047cb85652575) 中阅读 [本题报告](https://github.com/20171806011/EM-FloatFormer/blob/bbc76e3813a24840b9821c59cb8047cb85652575/research/stage3c/C/C_distillation.md) 和同目录代码、JSON。使用该提交内的模拟器、s17 权重及 8 比特缓存。以下命令从仓库根目录运行。

**研究问题**　固定观测、模型参数与候选打分器，推导精确后验作为教师时的梯度均值和协方差。换成从独立样本估计的有偏教师，求 n 个独立硬标签下使梯度均方误差最小的混合权重。独立枚举一个降低梯度误差却损害首选类别的例子，并检查教师使用了同一受评标签时推导中的哪一步失效。

```sh
python -B research/stage3c/C/check.py
```

**验收**　对照 results.json。报告可复核的公式、一个数值反例，以及在相同缓存、输入权限、训练预算与测评标签下怎样比较硬标签和教师监督。

**交付**　在独立分支 `research/stage3c-work-c` 的 `research/stage3c/work_results/C/` 保存简短报告、独立核验代码、结果和复现命令。写明假设、实跑样本数或枚举范围、复用的种子、与原结果吻合或相异之处，以及仍需宿主实验回答的问题。推送后回报提交链接和运行结果。
