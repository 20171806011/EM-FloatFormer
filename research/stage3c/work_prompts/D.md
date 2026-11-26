# Work 任务 D · 固定的 Q1/Q3 查找表是否选对每一类

你负责 EM-FloatFormer 的 D 题。请独立完成推导、执行数值核验，并将结果提交 GitHub。其他六组可以同时开展。

**起点**　在 [EM-FloatFormer 的冻结提交](https://github.com/20171806011/EM-FloatFormer/tree/bbc76e3813a24840b9821c59cb8047cb85652575) 中阅读 [本题报告](https://github.com/20171806011/EM-FloatFormer/blob/bbc76e3813a24840b9821c59cb8047cb85652575/research/stage3c/D/D_certificate.md) 和同目录代码、JSON。使用该提交内的模拟器、s17 权重及 8 比特缓存。以下命令从仓库根目录运行。

**研究问题**　审查已发布 s17 查找表在已知 C4 和无导频条件下的首选判决。推导为什么 16 个旋转轨道覆盖 256 对缓存；复算两阶段的独立母波形计数、最接近翻转的一行、96 项竞争类别的检验及全局 p 值。解释第一阶段同时置信界和第二阶段交并检验的错误预算。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B research/stage3c/D/check_d.py --repo .
```

**验收**　对照 results.json。列出样本数、独立单位、最弱计数、首选错误率或超额风险边界，并明确这份证书能支持哪些工况、哪些题以及哪一种损失。

**交付**　在独立分支 `research/stage3c-work-d` 的 `research/stage3c/work_results/D/` 保存简短报告、独立核验代码、结果和复现命令。写明假设、实跑样本数或枚举范围、复用的种子、与原结果吻合或相异之处，以及仍需宿主实验回答的问题。推送后回报提交链接和运行结果。
