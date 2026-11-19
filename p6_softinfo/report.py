"""Render measured evidence without modifying the frozen experiment."""
from datetime import datetime, timezone
import math
import statistics

from p6_softinfo.common import (BASE, CONDITIONS, MAIN_BASE, METHODS, RESULTS,
    ROOT, read, save, sha, verify_freeze)

NAMES = {"trust": "P6 Trust", "smooth": "前轮 Smooth", "variance": "仅方差",
         "profile": "仅相位精修", "variance_profile": "方差＋精修",
         "local_parity": "Local+Parity", "fixed_parity": "Fixed+Parity",
         "waveform_dd": "Waveform-DD", "lmmse_2d": "2D LMMSE"}
LABELS = {"intact": "完整导频", "first_attenuated": "首行×0.4",
          "first_erased": "首行擦除", "interior_erased": "内部导频行擦除（索引9）",
          "scattered_erased": "交错擦除一半导频"}


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                       "| " + " | ".join(["---"]*len(headers)) + " |",
                       *["| " + " | ".join(map(str, row)) + " |" for row in rows]])


def main():
    signature = verify_freeze()
    summary = read(RESULTS/"summary.json")
    numeric = read(RESULTS/"numerical_audit.json")
    tests = read(RESULTS/"tests.json")
    audit = read(RESULTS/"independent_audit.json")
    timing = read(RESULTS/"latency.json")
    if not (tests["passed"] and audit["passed"]):
        raise ValueError("Do not render a passed report for failed validation/audit")
    aggregate = {(row["condition"], row["method"]): row for row in summary["aggregates"]}
    core = METHODS[:5]
    blocks = table(["条件", *[NAMES[m] for m in core]], [
        [LABELS[c], *[f"{aggregate[c,m]['block_errors']}/5760 ({100*aggregate[c,m]['bler']:.3f}%)"
                      for m in core]] for c in CONDITIONS])
    ber = table(["条件", *[NAMES[m] for m in core]], [
        [LABELS[c], *[f"{100*aggregate[c,m]['ber']:.5f}%" for m in core]] for c in CONDITIONS])
    controls = table(["条件", *[NAMES[m] for m in ("trust", *METHODS[5:])]], [
        [LABELS[c], *[str(aggregate[c,m]["block_errors"]) for m in ("trust", *METHODS[5:])]]
        for c in CONDITIONS])
    primary = table(["条件", "比较基线", "BLER降低百分点", "点态95%区间", "胜/平/负", "原始p", "Holm p"], [
        [LABELS[r["condition"]], NAMES[r["baseline"]], f"{r['mean_bler_reduction_pp']:+.4f}",
         f"[{r['pointwise_t95_bler_reduction_pp'][0]:+.4f}, {r['pointwise_t95_bler_reduction_pp'][1]:+.4f}]",
         f"{r['wins']}/{r['ties']}/{r['losses']}", f"{r['p_raw']:.8f}", f"{r['p_holm']:.8f}"]
        for r in summary["primary_tests"]])
    numerical = table(["方法", "通过增益变换", "最大LLR绝对差", "最大CFR相对误差", "硬判决翻转"], [
        [NAMES[m], f"{r['passed_transformations']}/{r['gain_transformations']}",
         f"{r['max_grid_llr_diff']:.6g}", f"{r['max_relative_cfr_l2']:.6g}", str(r["coded_flips"])]
        for m,r in numeric["summary_by_mode"].items()])
    latency_map = {(r["batch_size"], r["method"]): r for r in timing["rows"]}
    latency = table(["Batch", *[NAMES[m] for m in core]], [
        [str(b), *[f"{latency_map[b,m]['median_ms']:.3f}" for m in core]] for b in (1,8,64)])
    mechanism = {m: dict(before=[], after=[], accepted=0, evaluations=0, pinv=0, invalid=0)
                 for m in ("profile", "variance_profile")}
    for i in range(1, 11):
        for row in read(RESULTS/"fresh"/f"pair{i:02d}.json")["rows"]:
            if row["method"] not in mechanism:
                continue
            p, dest = row["diagnostics"]["profile"], mechanism[row["method"]]
            dest["before"].extend(p["objective_before"])
            dest["after"].extend(p["objective_after"])
            for key, source in (("accepted", "accepted_steps"), ("evaluations", "candidate_evaluations"),
                                ("pinv", "used_pinv"), ("invalid", "invalid_curvature")):
                dest[key] += sum(p[source])
    mechanism_rows = []
    for method, values in mechanism.items():
        pairs = list(zip(values["before"], values["after"]))
        mechanism_rows.append([NAMES[method], str(len(pairs)), str(sum(a <= b for b,a in pairs)),
            f"{100*statistics.mean((b-a)/b for b,a in pairs):.6f}%", str(values["accepted"]),
            str(values["evaluations"]), str(values["pinv"]), str(values["invalid"])])
    mechanism_table = table(["方法", "burst×条件", "目标未增加", "平均相对下降", "接受步数", "候选评估", "伪逆回退", "异常曲率"], mechanism_rows)
    nmse_gain = aggregate["intact","trust"]["nmse_db"]-aggregate["intact","variance_profile"]["nmse_db"]
    sections = [
        "# P6 后续数学验证：条件方差与完整波形联合相位精修\n\n日期：2026-11-19。状态：**已实现、已实验验证；不推荐替换现有接收机。**",
        "## 1. 结论\n\n本次候选通过了数学恒等式、导数、集成和严格复增益数值检查，但没有证明译码性能胜出。"
        "完整导频下组合候选少错26/5760个块（BLER降低0.4514个百分点），预设检验不显著。"
        "在首行和内部行导频擦除下，组合候选比前轮Smooth分别多错141和178个块，BLER增加2.4479和3.0903个百分点；"
        "六项Holm校正后p分别为0.029296875和0.01171875，方向是候选退化。"
        "交错擦除条件另有明显的描述性退化：Trust/Smooth为1220个块，方差与组合方案均为1431个块，多错211个块。\n\n"
        "组合方案以原P6 Trust为底座；本轮没有测试Smooth再叠加新模块。因此这些结果说明该候选不能替代Smooth，"
        "不能表述为‘给Smooth加模块导致退化’。不改动原有默认模型、不宣称P7完成，也不把代码复杂度或阶段名称视作性能成果。",
        "## 2. 实际数学改动\n\n**条件方差：** 在原有16-QAM软均值之外计算条件方差，导频方差为零，保留原LDPC外信息。"
        "在独立频域符号协方差近似下，将完整跨射线的期望Gram修正加入线性R2波形幅度求解。"
        "该恒等式对所选近似是精确的；它没有恢复LDPC和ICI产生的完整后验相关性。\n\n"
        "**联合相位精修：** 最后一次幅度求解时固定软均值/方差，消去幅度，对三条射线共六个实相位坐标计算完整剖面目标的精确Hessian，"
        "保留跨射线项。采用正定移位、每步0.25弧度上限、两步上限与Armijo回溯。"
        "此局部优化计算显式分离输入；未宣称支持经过它的端到端训练。\n\n"
        "**仅有代数测试的工具：** 确定性符号与高斯幅度先验下、含log-det项的证据计算；归一化比特概率混合。"
        "它们没有进入本轮BER接收机，也没有产生经过验证的八模式多普勒后验。"
        "完整公式、数值规则和限制见[MATH.md](MATH.md)。",
        "## 3. 设计与公平性\n\nR2、9 dB、20×32二维OFDM、16-QAM、(128,64) LDPC，固定已有10组检查点pair06–15，"
        "不重新训练。每组4个B8新批次，共320个独立生成burst；同一burst在五种条件间重复使用。"
        "每种条件、每个方法均有5760个码块、737280个编码比特、368640个信息比特。"
        "新模拟种子为713611190000+1000×index+batch，与上轮后续研究不同。\n\n"
        "九个方法共享观测、导频、噪声信息和外部译码器；新方法沿用相同权重，未增加可训练参数。"
        "相位精修增加推理计算，因此没有计算预算相等的主张。消融包括仅方差、仅精修和组合。"
        "Local+Parity是匹配容量的局部对照，Fixed+Parity是固定物理求解器。"
        "经典Waveform-DD及2D LMMSE使用公开固定信道统计/结构先验，不接收真实信道。"
        "这些先验与原完整导频生成器相配，不与新增的未告知导频数字擦除过程匹配。\n\n"
        "首行乘0.4、首行零置、内部导频行（零起算索引9）零置、交错一半导频零置均保持导频掩码与报告噪声不变。"
        "这些是观测压力测试，不是未知信道家族泛化试验。"
        "[协议](PROTOCOL.md)与[冻结清单](FREEZE.json)在新样本评分前于本会话内固定，不是外部预注册。",
        "## 4. BER/BLER及经典对照\n\n### 核心方法：块错误数与BLER\n\n"+blocks+
        "\n\n### 原始编码比特BER（译码前）\n\n"+ber+
        "\n\n### 对照方法：块错误数（分母均为5760）\n\n"+controls+
        f"\n\n完整导频下，组合候选信道NMSE从{aggregate['intact','trust']['nmse_db']:.4f} dB降到"
        f"{aggregate['intact','variance_profile']['nmse_db']:.4f} dB，描述性改善{nmse_gain:.4f} dB；"
        "译码前BER却几乎不变，BLER改善不显著。信道估计误差改善不能替代通信错误率证据。"
        "所有信息比特BER、NMSE、逐批次精确计数及种子摘要均在[summary.json](results/summary.json)与[原始记录](results/fresh)。",
        "## 5. 预设六项统计比较\n\n表中正数表示组合候选比基线少错，负数表示组合候选更差。"
        "以10个检查点/新波形簇作为统计单位，枚举配对差值的1024种正负号翻转；"
        "要求零假设下簇效应符号可交换/对称。对六项预设BLER比较统一作Holm校正。"
        "点态t区间仅为十簇下的近似未校正区间。\n\n"+primary+
        "\n\n没有把单个LDPC块当作独立重复。其他条件、消融、BER和NMSE均为描述性结果。"
        "完整导频的非显著改善不是胜出；对Trust的不显著差异也不是非劣或等价证明。"
        "未依据新样本挑选另一个获胜版本。",
        "## 6. 目标函数下降为何不等于性能提升\n\n"+mechanism_table+
        "\n\n每个方法的1600项是320个burst×5种重复条件，并非1600个独立样本。"
        "这里核查的是各自固定后验近似下、包括方差项和原有ridge的物理波形目标。"
        "学习残差delta_h保留给后续解调，未加入这个物理拟合目标；最终LLR方差仍是原解调器近似。\n\n"
        "**解释性推断，尚未由新的消融确定因果：** 近似后验没有完整符号相关性；旧权重适应的是旧求解器；"
        "未告知的导频擦除破坏了当前观测模型；降低局部波形损失也可能强化错误模式。"
        "交错擦除中方差单项已经出现与组合相同的块错误增量，而仅精修接近原版，这提示应优先检查方差与观测模型的匹配。"
        "下一次可检验后验/输出LLR校准及在相同预算下重新训练的对照；本次没有执行这些未来实验。",
        "## 7. 结构、数值精度与测试\n\n31项新数学/集成测试与38项原P6回归测试全部通过。"
        "新恒等式通过独立有限星座枚举与显式Fourier协方差传播检查；幅度解对照增广最小二乘；"
        "梯度/Hessian对照有限差分，并检查跨射线块。关闭新改动的接收机复制在既定开发条件下与原版逐位一致。"
        "接收机只读取公开观测字段，检查点权重和输入不变。[测试收据](results/tests.json)。\n\n"
        "在三个复用开发检查点、十种退化/无噪声条件、27种复增益上审计，"
        "保留原严格LLR/CFR误差<1e-12且无硬判决翻转的标准：\n\n"+numerical+
        "\n\n复数连续样本波形、C*规范化和Hermitian注意力仍为原P6结构；本轮没有更换注意力。"
        "新变体各810/810次通过；四个方法共3240次。所有推理和本地Hessian均为CPU float64/complex128。"
        "这不是对所有输入的浮点定理，不是float32/混合精度或GPU验证。"
        "[完整数值记录](results/numerical_audit.json)。\n\n"
        "开发过程保留两个问题的记录：跨no_grad/inference_mode逐位相等的过强断言，以及冷启动时推理缓存张量进入自动微分的错误。"
        "前者仅调整跨上下文测试至既有1e-12数值标准（基线复制的逐位测试不变）；后者通过克隆缓存基底修复并增加冷缓存回归。"
        "二者均在冻结与新样本评分之前处理。[开发事件](results/development_incidents.json)。",
        "## 8. 延迟\n\n单位：每个batch的接收机推理毫秒中位数；含内部LDPC外信息和相位求导，"
        "不含模拟与最终外部LDPC译码。AMD EPYC 9V74、单Torch线程、2轮热身、10轮循环交错计时，"
        "计时期间无其他实验工作进程。\n\n"+latency+
        "\n\n原始时长和四分位数见[latency.json](results/latency.json)。相位精修增加延迟；"
        "没有稳定加速、实时部署或其他硬件上的结论。",
        "## 9. 复现与相对P4的边界\n\n另一代理在相同CPU/软件环境中独立重算全部45个条件/方法聚合的整数计数与六项统计，"
        "检查全部3200项精修目标记录，重新运行预选index1和10的完整九方法×五条件×四批次。"
        "[审计收据](results/independent_audit.json)区分整数精确一致与能量浮点容差，并保存输入文件指纹。"
        "它是同环境的独立代理/进程复核；没有外部实验室、第二套硬件或重新训练的独立复现。\n\n"
        "P4历史基准是[8380084c](https://github.com/20171806011/EM-FloatFormer/commit/8380084c6dece64119ab72f9c3c1c212e154fb17)。"
        "P4已经有二维OFDM、固定先验经典对照、正式统计及审计；P6已经有完整波形物理求解与Hermitian轴向算子。"
        "本轮新增的是上述两个数学模块与新的验证数据，并非重新发明这些结构。"
        "P4的正式10 dB强PA任务与本轮R2 9 dB线性多普勒任务不同，本轮未重跑P4模型，不能直接以跨阶段BLER数字宣称胜出。"
        "详见[历史P4报告](../p4/STAGE3_RESULTS.md)、[P6报告](../p6/P6_REPORT.md)与[前轮后续报告](../p6_followup/REPORT.md)。",
        f"## 10. 溯源\n\n- 公开main基准：`{MAIN_BASE}`。\n- 本研究父提交：`{BASE}`。\n"
        f"- 评分前冻结SHA-256：`{signature}`。\n"
        "- [所有新文件SHA-256](MANIFEST_SHA256.json)；历史目录未修改。\n"
        "- [运行入口与复算说明](README.md)。\n\n"
        "代码存在、执行过的实验、同环境独立重放和外部独立复现是不同证据等级；本轮只拥有前三项中上述明确列出的范围。",
    ]
    (ROOT/"REPORT.md").write_text("\n\n".join(sections)+"\n", encoding="utf-8")
    manifest = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(ROOT.rglob("*"))
                if p.is_file() and "__pycache__" not in p.parts
                and p.name != "MANIFEST_SHA256.json" and not p.name.endswith(".pyc")}
    save(ROOT/"MANIFEST_SHA256.json", dict(created_utc=datetime.now(timezone.utc).isoformat(),
         base_commit=BASE, freeze_sha256=signature, files=manifest))
    print(dict(report=str(ROOT/"REPORT.md"), manifest_files=len(manifest),
               fresh_grid_cells=10*4*5*9, freeze_sha256=signature), flush=True)


if __name__ == "__main__":
    main()
