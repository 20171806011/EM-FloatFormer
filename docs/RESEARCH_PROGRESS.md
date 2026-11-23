# EM-FloatFormer 研究进展

更新于 2026 年 11 月 23 日。**当前处于六阶段总计划的第 3 阶段前半。** 在民用合成四相信号上，一份先于提问生成的有限缓存，已经能支持两次追问：先读两个符号的相对相位，晚到的导频再帮助判断第一个符号。当前答案由专用读出程序给出。下一个里程碑是检验冻结表示在新条件下保留的信息，并完成首次 3B 宿主接入。[六阶段总计划](LONG_TERM_RESEARCH_PLAN.md) · [宿主接入计划](MULTIMODAL_INTEGRATION_PLAN.md)

| 阶段 | 截至本次更新的交付与待验问题 |
|---|---|
| 1．可辨识边界 | 四相专题已交付。理想模型中相对相位恰需 2 比特，两问都保证答对恰需 4 比特。已推导离散噪声下的逐题风险、两种时间间隔的影响，以及连续 I/Q 下有限缓存的误差界。[证明与复算](https://github.com/20171806011/EM-FloatFormer/blob/fa396e8f5041d9db1f71af60d7b1558f1370d7fb/research/fourphase_memory_pilot/README.md) |
| 2．配对判别实验 | 首套 CPU 合成基准已交付。保持题面和逐点幅度相同，转动第二窗或两窗同转，分别揭示有效、已知独立、重复或缺省参考，逐项比较答案概率。[基准与可复跑样例](https://github.com/20171806011/EM-FloatFormer/blob/6462a6ae04c2d96a72b70d69b69e1ba21f36896e/research/fourphase_paired_questions/README.md) |
| 3．共享编码器与首次模型接入 | 已交付 4、8 比特两窗缓存的手工表示、普通学习编码器和带四相旋转约束的学习编码器。下一项是冻结表示后的跨条件直读，以及宿主语言回答和直接读出的逐题对照。[冻结协议](https://github.com/20171806011/EM-FloatFormer/commit/9cc1b2516789a40c96d9bd204db40a58b50f61fb) · [同预算实验](https://github.com/20171806011/EM-FloatFormer/blob/504f3a440e737e290ba7e0fa62fd6667a04f5cde/research/matched_cache/README.md) |
| 4．独立采集验证 | 已有 Pluto、CTTC 与 435 MHz OFDM 接收研究。当前冻结编码器仍待用整场留出的真实采集验证其预测。[Pluto 报告](https://github.com/20171806011/EM-FloatFormer/blob/ade8333fb68126251a83e484b69b821b5d97bcea/research/realdata_pluto_qpsk/REPORT.md) · [CTTC 报告](https://github.com/20171806011/EM-FloatFormer/blob/1887aee6de9649bc0dd8a4f05fe3c4542f3f6d77/research/realdata_cttc_qpsk/REPORT.md) |
| 5．事件与关系 | 既有相位跳变、候选解释与证据更新研究提供了可复用的物理问题。让同一表示在新波形上定位事件、报告歧义并指出支持时间窗，仍待验证。[前期五组会合](https://github.com/20171806011/EM-FloatFormer/blob/e914befb2c03838c5e911e1af55c64230f1f42d6/research/abcde_convergence_2026_11_22/README.md) |
| 6．可追问的原生电磁模态 | 待交付多宿主真实录制演示与盲评。最终测试先编码、后揭题；能追加一次观测时，模型先选择参考窗并预测结果，再依据实际观测修订回答。 |

## 最近一轮改变了什么判断

在 1,024 条留出测试母波形、2 dB 条件下，手工表示和带旋转约束的学习编码器在两档缓存上都通过了三项检查：应转动的四项概率按角度平移，应保持的概率逐项保持，同一含噪导频重复呈现时置信度增量为零。普通编码器两档各三个种子均未通过全部检查。8 比特带约束模型在有效导频到来后的首符号准确率为 69.30%，手工表示为 68.95%；按母波形配对的差为 +0.36 个百分点（95% 区间 −0.49 至 +1.20）。现有数据支持保留物理旋转关系，准确率优势的方向仍不确定。重复观测的去重来自共享读出器和已知采样编号，自主识别参考来源仍待检验。[逐题结果与区间](https://github.com/20171806011/EM-FloatFormer/blob/504f3a440e737e290ba7e0fa62fd6667a04f5cde/research/matched_cache/results/summary.json)

## 第 3 阶段接下来的验收

1. 冻结已有编码器，改变参考位置与相位轨迹，测试未见问题组合。加入“参考揭示前预测留出观测”的题，先用解析模型给出可辨识条件，再比较有限缓存与完整 I/Q 的逐题风险。
2. 冻结编码器和 Ministral 3B 宿主，训练输入连接层。用相同的波形和参考权限对照直接读出与宿主答案，定位接入后丢失的关系。随后在 Qwen3.5-4B 上复验。

总计划中的分支清单记录 11 月 22 日五组会合时的仓库。本页列出此后新增的四相理论、配对基准和编码器比较，并链接到固定提交，供后续进展对照。
