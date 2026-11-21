# P12 两处一致性修复与验证记录

日期：2026-11-21。基线提交：[`0d01179d43bbf02a700ce1029e613d9af1b78c8f`](https://github.com/20171806011/EM-FloatFormer/commit/0d01179d43bbf02a700ce1029e613d9af1b78c8f)。分支：`research/p12-first-principles-diagnostics`。

本次完成离格接收器的两处一致性修复。全部 **18 项 P12 测试通过**。另用事先固定的 14 条新链路运行修复前后配对诊断：错块从 **124/252 增至 133/252**。这组诊断表现变差，不能作为性能提升或非劣的证据。本次没有据此调参，也没有重跑原 3,584 条正式种子实验。

## 1. 修复 B-anchor 后验回灌先验

原 `run_p12_offgrid_physics` 通过 `build_offgrid_branches` 获得已经吸收当前 B-anchor 观测的 `w_ray_classical`，再用它分配 inherit、clear、piecewise 三个轨迹 family 的先验质量。后续拟合又条件于同一 anchor，因此“Frame-A 先验／导频证据只计一次”的说明与实现不符。

修复后，完整 P12 路径直接从 Frame-A atoms 构造时延几何，完全不调用这个 B-conditioned bundle。family 质量采用事先固定、与观测和历史正式成绩无关的均匀分配：

| 可用 family | inherit | clear | piecewise |
| --- | ---: | ---: | ---: |
| 开启 piecewise | 1/3 | 1/3 | 1/3 |
| 关闭 piecewise | 1/2 | 1/2 | 0 |

family 内的已有模式先验保持原定义。基础先验与加入 AUX 分数后的有效先验均设为只读。每次 Jacobi refit 都从同一基础先验和同一个固定 AUX 分数重新计算，条件观测为去重后的 anchor 与已接纳 helpers，不沿用上一次 refit 的后验作为新先验；重复 known rows 会明确报错。历史 `B-refit off` 对照仍走原 P11 路径。

**保证范围：** 保留 Frame-A support 的混合分支不再把 B-anchor 后验作为轨迹 family 先验。B-only 时延发现、延迟重置分支的功率估计，以及基于 anchor 拟合矩的 AUX Gaussian quasi-likelihood 仍是 plug-in 近似。经验性 reset gate 仍可选择全 clear 分支。本修复不声称整个接收器已经成为严格的联合 Bayesian posterior，也不把原固定延迟阈值解释为推导得到的 BIC penalty。

## 2. 修复三折首轮的双侧 helper 保证

原分区仅保留循环 donor fold 中离 anchor 最近的一行。这个选择可能导致首轮两个 helper 位于同一侧；在 `anchor=16, probe=19, fold=1` 时，甚至把右侧唯一可用 helper 排入 AUX。

现在按原有“距离、行号”顺序扫描 donor，保留第一个能使接收器实际选出的最近两个 helper 分居 anchor 两侧的行。这个决定只读取公开行位置，不读取观测、真值或评测结果。原 AUX 已满足双侧条件时，选择保持不变。

原预算保持：目标行仍为交错的 **6+6+6**；每折 **5 个 AUX、7 个可用 helper**；三折 AUX 共 15 行且全局互斥；本折目标行和 probe 均不能进入本折拟合。上述已知反例修复后的首轮是 `[15, 18]`。

对全部 380 种不同 anchor/probe 布局逐一测试：300 种具备每侧至少两条公共数据行的布局，全部三折通过；其余 80 种因某侧数据不足而显式拒绝。公开 anchor 集合 `(3, 6, 10, 13, 16)` 的全部 15 折均通过。

## 3. 实际测试结果

`python -m pytest p12/tests -q` 的实际结果为 **18 passed，0 failed，0 errors，0 skipped，27.468 秒**。这是既有 11 项测试加新增 7 项；逐项执行记录见 [verification.json](results/consistency_fix_2026_11_21/verification.json)。

新增先验测试使用受控单径和固定 support，从独立的完整 `32×32` 复高斯协方差积分计算模式证据、信道均值与方差，核对实际 receiver 输出。两个 anchor 相位均匹配；故意把导频似然计两次的参考结果则可区分。此外，测试污染旧缓存门控权重，确认完整路径不调用旧 bundle；移除 bundle 后所需的观测合法性检查也得到保留。

新增几何测试覆盖全部 anchor/probe 布局、公开 anchor 的确定性、`anchor=16` 反例与非法输入。既有泄漏隔离、C* 不变性、Numba/NumPy 一致性、延迟重置和端到端 smoke 测试也全部通过。

## 4. 独立小规模配对诊断

协议在求值前固定为 9 dB、梳状导频 `(0,9,10,19)`、7 种场景各 2 条链路，共 14 条、每臂 252 个码块，覆盖五个公开 anchor。种子使用 `p12_e2e` 的新偏移 `20,000,000`，与原正式种子区间分离。两臂使用相同默认 receiver flags；没有按诊断结果选择参数、场景或种子。

基线在独立、无修改的 `0d01179` worktree 中运行，确保连同旧分区算法一并加载。修复版在修复后的工作树运行。两臂分别启动 Python 进程，记录实际导入源码、协议与 runner 的 SHA-256；**14/14 条链路的观测及真值哈希完全相同**。修复版输出中的 `git_head` 是运行时的基线 HEAD，具体未提交修复由 `imported_source_sha256` 唯一标识。

| 场景 | 每臂码块数 | 修复前错块 | 修复后错块 |
| --- | ---: | ---: | ---: |
| `offgrid_continuous` | 36 | 5 | 7 |
| `offgrid_partial_shift` | 36 | 14 | 15 |
| `offgrid_midframe_shift` | 36 | 14 | 15 |
| `offgrid_full_reset` | 36 | 14 | 19 |
| `offgrid_full_reset_new_delays` | 36 | 33 | 33 |
| `offgrid_ood_burst_and_shift` | 36 | 21 | 21 |
| `offgrid_ood_five_ray_fast` | 36 | 23 | 23 |
| **合计** | **252** | **124（49.2063%）** | **133（52.7778%）** |

修复版多 9 个错块，BLER 高 **3.5714 个百分点**。按每链路错块数计，0 条改善、10 条持平、4 条变差。该诊断同时包含两处修复，不能将差异归因于其中一处。252 个码块来自 14 条链路，不能把它们当成 252 个独立样本做显著性推断。本记录不作总体性能、显著性、优越性或非劣结论。

全部原始记录：

- [事先固定的协议](results/consistency_fix_2026_11_21/protocol.json)
- [修复前逐链路结果与源码指纹](results/consistency_fix_2026_11_21/baseline.json)
- [修复后逐链路结果与源码指纹](results/consistency_fix_2026_11_21/corrected.json)
- [测试、配对核对和历史文件完整性记录](results/consistency_fix_2026_11_21/verification.json)

## 5. 历史结果与复现

原 `p12/results` 的 **14 个文件逐字节保持不变**，包括原正式 JSONL、统计结果、审计和 `freeze_manifest.json`。`P12_REPORT.md` 仅增加版本说明，原表格保留。未修改 `p12/` 外的历史文件。

原冻结清单与 `0d01179` 基线一致；对修复版检查时，`offgrid_physics.py` 与 `offgrid_receiver.py` 两个源码哈希应当不匹配。这是防止旧实验身份被覆盖的预期行为，不能为了让检查通过而重建或覆盖原清单。原正式实验应在 `0d01179` 的独立 worktree 中复现。修复版的正式性能需要另行冻结协议并重新评测，本次尚未完成。

本次环境：Python 3.12.14、NumPy 1.26.4、SciPy 1.13.1、Numba 0.60.0、PyTorch 2.15.0+cpu、pytest 8.4.2；BLAS、OpenMP、Numba 与 PyTorch 均使用单线程设置，诊断启用 deterministic algorithms。以下在修复版仓库根目录、上述依赖环境中运行；输出写到临时目录，不覆盖本次已保存的记录：

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
python -m pytest p12/tests -q

git worktree add --detach ../EM-FloatFormer-p12-baseline 0d01179d43bbf02a700ce1029e613d9af1b78c8f

python p12/consistency_diagnostic.py \
  --repo-dir ../EM-FloatFormer-p12-baseline \
  --protocol p12/results/consistency_fix_2026_11_21/protocol.json \
  --label baseline --output /tmp/p12-consistency-baseline.json

python p12/consistency_diagnostic.py \
  --repo-dir . \
  --protocol p12/results/consistency_fix_2026_11_21/protocol.json \
  --label corrected --output /tmp/p12-consistency-corrected.json
```

计时仅为该环境中的运行记录，不构成硬件或速度基准。本次工作止于这两处修复、验证和版本记录。
