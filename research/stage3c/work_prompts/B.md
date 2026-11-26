# Work 任务 B · 哪些相位与时间信息足以确定预测

你负责 EM-FloatFormer 的 B 题。请独立完成推导、执行数值核验，并将结果提交 GitHub。其他六组可以同时开展。

**起点**　在 [EM-FloatFormer 的冻结提交](https://github.com/20171806011/EM-FloatFormer/tree/bbc76e3813a24840b9821c59cb8047cb85652575) 中阅读 [本题报告](https://github.com/20171806011/EM-FloatFormer/blob/bbc76e3813a24840b9821c59cb8047cb85652575/research/stage3c/B/B_features.md) 和同目录代码、JSON。使用该提交内的模拟器、s17 权重及 8 比特缓存。以下命令从仓库根目录运行。

**研究问题**　假设公共频偏恒定、相位沿同一条布朗轨迹扩散。对任意有标签的窗口、预测与导频时刻，推导相量与累计方差如何确定联合相位律。构造只给终点转角和只给各时刻相量各自会混淆的两种物理情形。另在声明先验后比较已知和未知频偏符号，写明接收者是否知道初始朝向。

```sh
python -B research/stage3c/B/check_b.py
```

**验收**　对照 results.json，用第二种计算方法核验至少一个反例；最后列出正式实验两臂共用的元数据、只在相位臂出现的数值特征及成立条件。

**交付**　在独立分支 `research/stage3c-work-b` 的 `research/stage3c/work_results/B/` 保存简短报告、独立核验代码、结果和复现命令。写明假设、实跑样本数或枚举范围、复用的种子、与原结果吻合或相异之处，以及仍需宿主实验回答的问题。推送后回报提交链接和运行结果。
