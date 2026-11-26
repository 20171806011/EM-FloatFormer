# Work 任务 F · 怎样判断宿主真的使用了频偏符号

你负责 EM-FloatFormer 的 F 题。请独立完成推导、执行数值核验，并将结果提交 GitHub。其他六组可以同时开展。

**起点**　在 [EM-FloatFormer 的冻结提交](https://github.com/20171806011/EM-FloatFormer/tree/bbc76e3813a24840b9821c59cb8047cb85652575) 中阅读 [本题报告](https://github.com/20171806011/EM-FloatFormer/blob/bbc76e3813a24840b9821c59cb8047cb85652575/research/stage3c/F/F_signed_host_probe.md) 和同目录代码、JSON。使用该提交内的模拟器、s17 权重及 8 比特缓存。以下命令从仓库根目录运行。

**研究问题**　分别生成真实频偏为 ±0.35 的配对世界，共享原始符号、布朗创新和噪声，同时重算各自的缓存与真实标签。在每个世界固定缓存和标签，再单独交换告知读出器的符号。保留四类概率，比较匹配与交换的对数损失，计算响应方向与幅度。构造首选答案逐条相同、对符号的概率响应却不同的两种读出器。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B research/stage3c/F/check_f.py --repo .
```

**验收**　对照 results.json，报告同缓存符号干预的配对区间、一个直接按样本对数比复算的结果，以及冻结宿主应导出哪些候选分数才能实际判定其行为。

**交付**　在独立分支 `research/stage3c-work-f` 的 `research/stage3c/work_results/F/` 保存简短报告、独立核验代码、结果和复现命令。写明假设、实跑样本数或枚举范围、复用的种子、与原结果吻合或相异之处，以及仍需宿主实验回答的问题。推送后回报提交链接和运行结果。
