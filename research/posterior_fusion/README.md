# EM-FloatFormer 研究二 · 后验融合

从 `REPORT.pdf` 阅读按八项交付组织的完整报告。`pages/` 是可编辑的中文源文，`audit/source_blobs.json` 记录指定提交中实际查阅的源码对象。包内文件集中在 `research/posterior_fusion/`，可独立使用。

## 数值复现

在本目录运行。Python 3.9 或更高版本，验证及计时均只用标准库。

```bash
python3 validation/validate.py --out /tmp/emf2-results.json
cmp /tmp/emf2-results.json validation/results.json
python3 validation/crosscheck.py
```

第一条命令用冻结协议重建三个训练集并重训参数，对新独立链路的有限总体做精确求和。第二条命令比对全部结果。第三条命令独立枚举五个读数的噪声翻转，用保存的可见训练频数重建学习权重，再核对六行风险。`validation/protocol.json`、`protocol.sha256`、`results.json` 和 `timings.json` 分别保存冻结设定、协议指纹、确定性实验结果及运行环境相关成本。

若需重新计时，在 `validate.py` 的命令末尾添加 `--timing /tmp/emf2-timings.json`。两行 finite-context 权重均可缓存，原始计时对前三类策略查表，对 attention 现场计算 softmax。

## 重建 PDF

源文可直接按 `pages/01.md` 至 `pages/12.md` 顺序阅读。使用安装有 Python、Pandoc、TeX Live/XeLaTeX、Latin Modern Math、Noto Serif 和 Noto Serif/Sans CJK SC 的环境，可以运行：

```bash
python3 layout/render_report.py pages /tmp/emf2-report.pdf
```

## 核查身份

审查固定在 [提交 655de6a](https://github.com/20171806011/EM-FloatFormer/commit/655de6aa42a08fe73e4f67f7060fb667d95518f1)。在已有包含该提交的本地 Git 克隆中，可用 `git rev-parse '655de6aa42a08fe73e4f67f7060fb667d95518f1:p12/offgrid_receiver.py'` 与 `audit/source_blobs.json` 中对应 `git_blob` 比对，其余文件同理。包内 `SHA256SUMS` 校验本研究文件，与原 P12 冻结清单独立。
