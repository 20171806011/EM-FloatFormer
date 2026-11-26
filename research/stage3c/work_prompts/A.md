# Work 任务 A · 迟到的导频能提供多少新信息

你负责 EM-FloatFormer 的 A 题。请独立完成推导、执行数值核验，并将结果提交 GitHub。其他六组可以同时开展。

**起点**　在 [EM-FloatFormer 的冻结提交](https://github.com/20171806011/EM-FloatFormer/tree/bbc76e3813a24840b9821c59cb8047cb85652575) 中阅读 [本题报告](https://github.com/20171806011/EM-FloatFormer/blob/bbc76e3813a24840b9821c59cb8047cb85652575/research/stage3c/A/Q2.md) 和同目录代码、JSON。使用该提交内的模拟器、s17 权重及 8 比特缓存。以下命令从仓库根目录运行。

**研究问题**　从冻结的两个 4 比特窗口索引出发，推导第一发送符号 Q2 在一次有效导频、同相位的两次独立采样、同一采样副本、独立无关参考下的四类后验。明确在哪个潜变量上先乘似然再积分。给出一个算例，把这一步与分别积分后相乘作比较，并说明采集编号和相位时刻各自起什么作用。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B research/stage3c/A/check_q2.py --repo . --export-kernel /tmp/stage3c-A-kernel.json
```

**验收**　对照 q2_results.json 与 q2_s17_kernel.json。分别报告增加真实观测和只精化旧读出的配对准确率、对数损失和不确定度，交付一个可直接用于模型评测的参考读出规则。

**交付**　在独立分支 `research/stage3c-work-a` 的 `research/stage3c/work_results/A/` 保存简短报告、独立核验代码、结果和复现命令。写明假设、实跑样本数或枚举范围、复用的种子、与原结果吻合或相异之处，以及仍需宿主实验回答的问题。推送后回报提交链接和运行结果。
