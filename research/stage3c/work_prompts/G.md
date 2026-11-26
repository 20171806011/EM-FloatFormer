# Work 任务 G · 一次模型读出能保证哪一种旋转对称性

你负责 EM-FloatFormer 的 G 题。请独立完成推导、执行数值核验，并将结果提交 GitHub。其他六组可以同时开展。

**起点**　在 [EM-FloatFormer 的冻结提交](https://github.com/20171806011/EM-FloatFormer/tree/bbc76e3813a24840b9821c59cb8047cb85652575) 中阅读 [本题报告](https://github.com/20171806011/EM-FloatFormer/blob/bbc76e3813a24840b9821c59cb8047cb85652575/research/stage3c/G/README.md) 和同目录代码、JSON。使用该提交内的模拟器、s17 权重及 8 比特缓存。以下命令从仓库根目录运行。

**研究问题**　对两个窗口分别旋转 a、b 个四分之一圈。由缓存码 4r+i 推导 Q1、Q2 和重发 Q3 的类别变换，构造先化到代表元、调用一次读出器、再把概率变换回去的算法。证明它对称，分析把 16 个旋转预测对齐再平均时总体风险下降所需的分布和损失条件。构造一个首选准确率或单次代表元风险变差的反例。

```sh
python -B research/stage3c/G/check_g.py research/stage3b_math/A/c4_s17_lookup.json
```

**验收**　对照 results.json，核验全部 256 个缓存对、16 个旋转和三道题。分别给出普通与代表元读出的最多调用数，说明固定元数据、连续导频与离线评分对这个计数的影响。

**交付**　在独立分支 `research/stage3c-work-g` 的 `research/stage3c/work_results/G/` 保存简短报告、独立核验代码、结果和复现命令。写明假设、实跑样本数或枚举范围、复用的种子、与原结果吻合或相异之处，以及仍需宿主实验回答的问题。推送后回报提交链接和运行结果。
