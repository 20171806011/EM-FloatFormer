# em-floatformer-e　冻结门控实验与配对计分器

固定审计提交 `5f88a5be984493bcb6aa164d84253b6f29876eef`。本包不依赖第三方模块。B 可将整个 em-floatformer-e 目录纳入仓库；代码不访问网络，不下载 OTA 数据，不训练接收机。数学证明及反例见 THEORY.md，面向统一前端的冻结方案见 PROTOCOL.md。

## 运行
Node.js 18 或更新版本，在包目录执行：

    node selftest.cjs
    node cli.cjs fit fixtures/plan.json fixtures/dev.json my-new.lock.json
    node cli.cjs score fixtures/plan.json my-new.lock.json be91fd79c4e788d0827c38ac7f5c71a90f1982cf704a6188c5df52fc6c2ff2da fixtures/test.json

该哈希仅对应随包的合成输入和代码；参考锁位于 fixtures/synthetic.lock.json，参考评分在 expected/synthetic-score.json。fit 独占创建新锁，已有同名文件会退出。正式数据用 fit 输出中事先登记、独立保存的 lock_sha256；勿从待评分文件临时计算“可信”哈希。锁证实版本一致，不替代保管未揭盲数据。CLI 成功返回 JSON，错误写 stderr 并以 1 退出。无需拟合即可用参考锁和相同哈希重放合成输入。

## 输入
- fixtures/plan.json 给出完整可运行模板。alg.frontend/static/dynamic 为版本的 64 位小写十六进制 SHA-256。合成例中的 a/b/c 重复串是哨兵，正式计划必须替换为实际产物摘要并归档生成清单。raw 固定 C0_raw_ota（含自然异常），conditions 每个条目显式给正 weight 和主 rep；权重和为 1。bits、block、block_origin、guard 和 margin 均显式给出。
- ranges 是有 session、lo、hi、phase 的闭区间数组。同一物理录制只分配一个稳定 ID。fit、select、test 都需登记。映射出的宿主全集就是期望观测清单，不能把缺失主帧等同零误码。design 为 historical 或 prospective_independent；后者须附采样代表性和独立性依据，外部还需核实来源、重复录制及固定会话采样配额。
- 开发输入与测试输入是两个独立 JSON 对象，均仅含 alg 和 rows。每个 row 的字段恰为 session、tti、condition、rep、z、truth、S、D。z 恰含 residual、disagreement，允许 null 作退回 S 的标志。truth 是定长 0/1 串，S 和 D 是同长度预测位串，? 表示擦除。三者使用相同载荷位置。映射器须保证相同扰动、对齐和码字；同宿主真值必须相同。实例身份 (session,tti,condition,rep) 唯一。
- 开发期 fit 只见两段 Dev，锁定中位数、门值和 SHA 后，score 仅收 Test。score 中两路预测同时存在用于盲评后的配对计分；gate.decision(model,z) 不接收 truth、条件、ID 或任一路最终比特。产生 z 的接收侧实现和耗时仍须随统一前端核验。

## 输出
- audit 保存 plan、dev、test、engine、lock 的哈希。byFrame 给每个主实例的宿主、块、门值、错数及对两条固定基线的 U/H。byBlock、bySession、strata、pooled 给位数 bits、eS/eD/eG、uS/hS/netS、uD/hD/netD 及三路 BER。pooled 是场景位整数计数；weighted 按预设混合计算，其 dS/dD 正为门有利。weighted.vsBest 是两个差的最小值。
- raw.harm 原始层门 BER 减静态 BER，正为有害。raw.pointWithin 是点估计，raw.upper 和 raw.noninferiority 才体现推断。inference.status 的 PASS 需要两条总体基线都被严格击败且原始非劣成立。DESCRIPTIVE_ONLY 不提供总体非劣结论。输出 primaryRows、technicalExcluded、uniqueHosts、sessions 和 blocks，层数、技术重复及位数均不增加独立会话数。

## 验证边界
精确枚举与 gate.cjs/selftest.cjs 已在 V8 运行，77 次断言通过，包含 10 个拒收输入。另以虚拟文件系统运行 cli.cjs 的相同参数分支，核对合成锁定、评分、错误锁哈希、锁覆盖、计划变动和计分核心变动。expected/verification.json 记录这些结果。移交前能回读载体并验证文件摘要。真实 Node 进程、操作系统 I/O、Python 解包器进程、实际统一前端适配和 240 帧回放本轮均未在宿主环境执行；没有 OTA 拟合门限或实测通过结论。示例里的两个 TTI 只是反例 B 的独立宿主标识，来自同一合成会话。不得据此推断独立部署总体。
