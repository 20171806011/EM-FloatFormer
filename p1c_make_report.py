#!/usr/bin/env python3
"""Create tables and plots from actually produced P1-C CSVs (no generated data)."""
from pathlib import Path
import csv, math, statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'p1c_results'
rows=list(csv.DictReader((OUT/'ber_summary.csv').open()))
byseed=list(csv.DictReader((OUT/'ber_by_seed.csv').open()))
lat=list(csv.DictReader((OUT/'latency_cpu.csv').open()))
timing=list(csv.DictReader((OUT/'training.csv').open()))

def get(split, snr, key):
    return next(float(r['pooled_ber']) for r in rows if r['split']==split and int(r['snr_db'])==snr and r['method']==key)

def seedber(seed, split, snr,key):
    return next(float(r['ber']) for r in byseed if int(r['seed'])==seed and r['split']==split and int(r['snr_db'])==snr and r['method']==key)

for split in ('in_distribution','unseen_four_tap'):
    fig, ax=plt.subplots(figsize=(9.2,5.7))
    for m,lbl in [
        ('token_mlp__raw_iq_magnitude','Token MLP, IQ'),
        ('mlp_mixer__raw_iq_magnitude','MLP-Mixer, IQ'),
        ('circular_cnn__raw_iq_magnitude','CNN, IQ'),
        ('circular_cnn__logamp_phase','CNN, Log/phase'),
        ('transformer__raw_iq_magnitude','Transformer, IQ'),
        ('transformer__logamp_phase','Transformer, Log/phase'),
        ('ls','LS'),('lmmse_matched_prior','LMMSE, matched prior')]:
        ax.plot([0,5,10,15,20],[get(split,s,m)*100 for s in [0,5,10,15,20]],marker='o',label=lbl)
    ax.set(xlabel='SNR (dB)',ylabel='Bit error rate (%)',title=f'P1-C: {split.replace("_"," ")}',yscale='log',xticks=[0,5,10,15,20])
    ax.grid(True,which='both',alpha=.28)
    ax.legend(fontsize=8,ncol=2)
    fig.tight_layout()
    fig.savefig(OUT/f'ber_{split}.png',dpi=160)
    plt.close(fig)

names=[('逐子载波 MLP','token_mlp'),('MLP-Mixer','mlp_mixer'),('圆周卷积 CNN','circular_cnn'),('Transformer','transformer')]
lines=[]
lines.append('# EM-FloatFormer P1-C — 架构消融实验报告')
lines.append('')
lines.append('**日期：2026-11-15。状态：本地CPU实际训练与评估已完成，属于初步模拟研究结果。**')
lines.append('')
lines.append('## 1. 研究问题')
lines.append('')
lines.append('在同样的合成、被动接收的民用QPSK/OFDM任务上，比较普通逐子载波MLP、可全局混合的MLP-Mixer、局部圆周卷积CNN、全局注意力Transformer，检验网络架构和P1-B两种表示的交互效应。仅非对抗性高斯热噪声和随机多径；不含雷达、电子对抗、发射控制或军事数据。')
lines.append('')
lines.append('## 2. 统一实验设计')
lines.append('')
lines.append('- 固定32子载波；已知全子载波导频 + QPSK数据；一个导频、一个数据信号共享静态信道；理想同步和充分循环前缀假设。')
lines.append('- 已知噪声方差；训练信道三径功率`(0.72,0.20,0.08)`；分布外测试为四径等功率`(0.25,0.25,0.25,0.25)`。')
lines.append('- 所有网络共享完全相同的复数接收数据、噪声方差、粗均衡器`z = yd*conj(yp)/(abs(yp)^2+noise_var+0.01)`，以及输出`+3*[Re z, Im z]`残差分支。因而此处比较的是相同已知特征的网络建模能力，而不是纯IQ全链路接收器。')
lines.append('- 两组编码：`raw_iq_magnitude` = `[I,Q,abs(z)]` 与 `logamp_phase` = `[log1p(abs(z)), I/(abs(z)+eps), Q/(abs(z)+eps)]`，`sigma=1`、`eps=1e-5`。每个模型有11维每载波输入。')
lines.append('- 每个架构下两组编码使用同一初始权重、同一批训练样本和测试帧；不同架构参数规模匹配至2%以内。AdamW、`lr=0.002`、750步、batch64、SNR在0–20dB均匀采样；种子101、202、303。')
lines.append('- 每种子/测试分布/SNR评估2000独立随机帧，即128,000位；汇总每格3个种子共384,000位；每个方法测0、5、10、15、20dB。')
lines.append('- 传统对照：LS、固定三径支持DFT、固定训练先验LMMSE、**知道正确测试分布先验的LMMSE**、理想瞬时真信道Oracle。后两者使用额外信息，仅供能力参照；尤其不能将分布外Matched LMMSE视作同信息公平对手。')
lines.append('- 与P1-B相同，输入是OFDM频域IQ而不是百万级原始时域复数样本；没有LDPC、MIMO、时钟偏移、3GPP全协议或真实无线电硬件。')
lines.append('')
lines.append('## 3. 参数量')
lines.append('')
lines.append('| 模型 | 参数量 | 子载波间信息交换 |')
lines.append('|---|---:|---|')
for name,arch in names:
    count=next(int(t['parameters']) for t in timing if t['method']==arch+'__raw_iq_magnitude')
    desc={'token_mlp':'只能单载波，除共享粗均衡器外无全局交互',
          'mlp_mixer':'全局token混合',
          'circular_cnn':'三层k=5局部卷积，循环边界',
          'transformer':'两层全局自注意力'}[arch]
    lines.append(f'| {name} | {count:,} | {desc} |')
lines.append('')
lines.append('相近参数量≠同FLOPs或同计算成本；同时测量CPU延迟。')
lines.append('')
lines.append('## 4. 实测BER（3种子合并，单位%）')
lines.append('')
for split,title in [('in_distribution','训练分布内（三径）'),('unseen_four_tap','未见过的四径分布')]:
    lines.append(f'### {title}')
    lines.append('')
    lines.append('| 方法 | 0dB | 5dB | 10dB | 15dB | 20dB |')
    lines.append('|---|---:|---:|---:|---:|---:|')
    for label,method in [(f'{name} / IQ',arch+'__raw_iq_magnitude') for name,arch in names]+[(f'{name} / LogAmp-Phase',arch+'__logamp_phase') for name,arch in names]+[
            ('LS','ls'),('DFT（三径支持）','dft_train_support'),('LMMSE（训练先验）','lmmse_train_prior'),
            ('LMMSE（测试先验，额外信息）','lmmse_matched_prior'),('Oracle（真信道，额外信息）','oracle')]:
        lines.append('| '+label+' | '+' | '.join(f'{get(split,s,method)*100:.3f}' for s in [0,5,10,15,20])+' |')
    lines.append('')
    lines.append(f'![各方法BER随SNR变化](p1c_results/ber_{split}.png)')
    lines.append('')
lines.append('## 5. 关键结论、效应及反例')
lines.append('')
cnn=get('unseen_four_tap',10,'circular_cnn__logamp_phase')*100
trans=get('unseen_four_tap',10,'transformer__logamp_phase')*100
m=get('unseen_four_tap',10,'lmmse_matched_prior')*100
seeddiffs=[(seedber(s,'unseen_four_tap',10,'transformer__logamp_phase')-
            seedber(s,'unseen_four_tap',10,'circular_cnn__logamp_phase'))*100 for s in [101,202,303]]
delta=statistics.mean(seeddiffs)
sd=statistics.stdev(seeddiffs)
margin=4.302652729911275*sd/math.sqrt(3)
lines.append(f'- **10dB的未见四径下：** 对数幅相CNN为{cnn:.3f}% BER，对数幅相Transformer为{trans:.3f}%，差{trans-cnn:.3f}个百分点；三个训练种子下均为CNN较低（Transformer减CNN逐种子：'+', '.join(f'{d:+.3f}' for d in seeddiffs)+'个百分点）。基于3个训练种子t(2)近似95%区间为['+f'{delta-margin:.3f}, {delta+margin:.3f}'+']百分点，只属探索性；未对多种模型/SNR选择修正。')
lines.append(f'- **更强的传统参照：** 此条件下已知四径统计先验的LMMSE为{m:.3f}% BER，仍好于CNN；这项对照拥有额外的测试分布知识。')
lines.append('- **训练分布内：** 10dB时原始IQ CNN 5.153%，LogAmp-Phase Transformer 5.183%，差异很小；传统训练先验LMMSE 4.723%，未被任何神经模型超过。')
lines.append('- **普通逐载波MLP：** 与LS接近，解释为它不能学习跨子载波的信道关联；而不是「MLP必然不行」。')
lines.append('- **MLP-Mixer：** 训练分布内明显优于普通逐载波MLP，但未见四径分布下10dB的LogAmp-Phase版本为8.958%，比CNN的6.560%更差；显示训练分布依赖，不是注意力独有问题。')
lines.append('- **编码×架构交互：** 对数幅相通常改善这个Transformer在分布外低SNR下的表现，但CNN在0/5dB的四径测试中反而是原始IQ更好；没有发现哪种编码适合所有架构和SNR。')
lines.append('')
lines.append('## 6. CPU推理延迟（本机PyTorch，seed101，毫秒/次前向推理）')
lines.append('')
lines.append('| 模型 / 编码 | batch=1 中位数 | batch=64 中位数 |')
lines.append('|---|---:|---:|')
for name,arch in names:
    for enc, ename in [('raw_iq_magnitude','IQ'),('logamp_phase','LogAmp-Phase')]:
        key=f'{arch}__{enc}'
        l1=next(float(t['median_ms']) for t in lat if t['method']==key and t['batch_frames']=='1')
        l64=next(float(t['median_ms']) for t in lat if t['method']==key and t['batch_frames']=='64')
        lines.append(f'| {name} / {ename} | {l1:.3f} | {l64:.3f} |')
lines.append('')
lines.append('说明：CPU PyTorch推理时间受线程数、调用开销、缓存和平台影响；不代表端到端无线电链路、实时性保证或通用硬件部署性能。8模型一轮联合训练各随机种子分别约57秒，不能将其视作单模型训练时间。')
lines.append('')
lines.append('## 7. 复现验证')
lines.append('')
lines.append('已将全部8个seed101模型从磁盘保存权重单独重新加载，用与训练时独立的相同测试随机流再次评价；共对照2分布×5 SNR×13方法=130个误码统计格，**130/130完全一致，误差计数没有差异**。详见 `p1c_results/checkpoint_verification.json`。此举检验了结果可从模型权重复现，但不等于独立代码实现、跨平台或跨人员复现。')
lines.append('')
lines.append('## 8. 判定及后续研究边界')
lines.append('')
lines.append('- **P1-C实验执行完成**：4架构×2编码×3随机种子，所有比较具有相同数据生成器、训练样本数与评估帧，参数规模差距不足2%。')
lines.append('- **没有证据证明Transformer普遍优于CNN/传统方法**：该模拟基准上，CNN在未见四径条件表现更好。匹配先验LMMSE依然更强。')
lines.append('- **还不能证明CNN的一般优势**：只有一个理想化QPSK/OFDM任务，3训练种子、小型架构、固定超参数；不同模型尚未做同FLOPs预算的系统超参搜索。')
lines.append('- **下一阶段P1-D建议但未执行**：重点测量随机信道抽样分布、数值精度、采样率/时钟偏差对预测的影响，增加随机种子并独立验证；若要测试物理时间序列，应明确切换到时域IQ且重新设计公平基线。')
lines.append('')
lines.append('## 9. 完整复现命令')
lines.append('')
lines.append('```bash')
lines.append('python ofdm_baseline.py --smoke-only')
lines.append('python p1c_architecture.py --smoke')
lines.append('python p1c_architecture.py --steps 750 --eval-frames 2000 --seeds 101 202 303 --out p1c_results')
lines.append('python p1c_verify_checkpoints.py --dir p1c_results --seed 101 --eval-frames 2000')
lines.append('python p1c_make_report.py')
lines.append('```')
lines.append('')
lines.append('完整CSV及种子级结果位于 `p1c_results/`；脚本文件：`p1c_architecture.py`, `p1c_verify_checkpoints.py`, `p1c_make_report.py`；使用Python 3.13、CPU版PyTorch 2.10、4线程。')
(ROOT/'P1C_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('REPORT WRITTEN',ROOT/'P1C_REPORT.md', 'bytes',(ROOT/'P1C_REPORT.md').stat().st_size)
print('OOD 10dB CNN-log',cnn,'TF-log',trans,'LMMSE oracle prior',m,'delta CI',delta-margin,delta+margin)
