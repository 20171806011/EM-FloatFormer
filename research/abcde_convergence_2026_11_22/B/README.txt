任务 B · EM-FloatFormer 逐目标审计

REPORT.md 为中文报告。mini.json、fullgrid.json 是已实际执行的独立 JS 输出。

node mini.js
node fullgrid.js

Python 对照要求 Python 3.10+、NumPy、SciPy、PyTorch。若没有两份固定提交，可执行：
python fetch_pins.py
python audit.py --ofdm pins/ofdm --pluto pins/pluto --crosscheck-js > synthetic-source.json

已有数据时，传一帧 stress 后 NPZ：
python audit.py --ofdm pins/ofdm --frame one-stressed-tti.npz --target 7 45 --tau 2 --stay .88

提交和 git blob 清单见 source-integrity.json、fetch_pins.py。脚本只按一个目标回放一帧；CLI 无训练/多帧运行入口。
