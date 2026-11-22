# EM-FloatFormer D 相位跳变理论对照

固定提交 `5f88a5be984493bcb6aa164d84253b6f29876eef`。本目录独立于训练和实测回放，无第三方依赖。在 Node.js 中从本目录执行：

```sh
node enumerate.cjs > regenerated.json
node -e "require('node:assert/strict').deepEqual(require('./regenerated.json'), require('./results.json'))"
node verify.cjs
```

* `PROOF.md` 给出固定源码审计、等价状态与比特表、已知和未知跳点的证明，以及何时能把这些数值当作下界。
* `PROTOCOL.md` 固定四个独立信息条件，区分增加一个导频与替换一个数据 RE，另列未来 OTA 对照。
* `enumerate.cjs` 用精确整数系数分组、按逐位多数计数，向标准输出写 JSON。
* `results.json` 是上述输出。BER 以最简有理数字符串保存。每组每比特计数分母 627200 = 28 个状态 × 16 个待测符号 × 1400 个 RE，是理想先验的穷举权重，不是实测样本数。
* `verify.cjs` 仅为比较结果而读取 JSON；使用独立的复旋转、星座逆查和导频约束推断候选状态，并逐项比对计数。浮点容差为 1e-8，主枚举不使用浮点容差判断等价。

交付验证在 JavaScript V8 中运行。Node 命令为可移植入口，本轮没有在本地 Node 进程或 OTA 数据上运行。
