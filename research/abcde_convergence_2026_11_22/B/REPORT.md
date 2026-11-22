# OFDM 与 PLUTO 两套接收机的逐目标审计

OFDM 固定提交 `5f88a5be984493bcb6aa164d84253b6f29876eef`。PLUTO 固定提交 `ade8333fb68126251a83e484b69b821b5d97bcea`。所有索引从 0 开始。对 OFDM，目标是施加已选 stress 后网格上的一个载荷 RE，删除发生在接收机入口。施加扰动是测量生成过程。若两个反事实从原始 IQ 起重新生成依赖整帧功率的噪声，其他 RE 也可能随之改变，本报告的探针不作这种干预。[扰动边界](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/stress_channel.py#L81-L115)

**结论**　OFDM B3 用含目标的平滑状态权重乘目标的原始 16-QAM 似然。按它所声称的固定参数生成模型，目标局部证据因此重复进入有效候选赔率。PLUTO B3b 用含目标的整帧权重乘已归一化的分支内符号后验，目标边际项在分支内抵消。上述判断与是否使用真值标签是两个问题。在给定 PLUTO 拟合候选库的条件下可以严格证明解调恒等式；端到端的物理建模还含经验估计。Group C 的相位、方差和门控均可能依赖目标，属于利用整帧上下文训练的条件预测器。

# 从量到判决

|量|实际输入|含当前目标 IQ？|如何进入最终判决及固定提交行号|
|---|---|---|---|
|OFDM 坐标、标签位序|行 0–13，FFT 13–114 去 DC 64，导频行为 2。载荷掩码总是去掉 14 个原始导频，包括可见导频变少的条件。自然二进制 I 高两位、Q 低两位|坐标不含，抽取值含|1400 个 RE 按行读出；正 LLR 判 1，等于 0 判 0。[几何](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/ofdm_data.py#L31-L49)、[映射/掩码](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/ofdm_data.py#L78-L99)、[读出](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/group_b_posterior_fusion.py#L127-L145)、[硬判](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/group_b_posterior_fusion.py#L171-L178)|
|OFDM 候选 H|当前可见导频 LS 插值；历史库按训练段 SINR 选参考，只存导频信道；当前导频拟合复增益、按残差加权并收缩|否|0 静态 LS；1–8 斜率；9–15 相跳在所有 s≠2 生效，亦含导频前行；16 内容寻址。影响每点密度和 B1/B2 点估计。[LS](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/ofdm_data.py#L269-L275)、[参考与对齐](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/hypothesis_frontend.py#L75-L137)、[候选编号](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/hypothesis_frontend.py#L220-L241)|
|OFDM 基线方差|远保护带均能、导频二阶差分、导频行 87 个数据 RE 最近点残差的截尾均值、信道功率下界和全部活跃 RE 均能上限|仅 s=2 载荷参加截尾；任意目标参加功率上限。保护带、导频差分、信道功率不含|候选共用的全帧拟合量。max/min、截尾会使某次删除的数值变化为零。[估噪](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/ofdm_data.py#L289-L315)|
|OFDM NBI 局部方差/超额能量|活跃载波压缩索引上的 3×3 IQ 能量，edge 复制边界；除以 LS 信号能量加基线方差；阈值 1.75，方差有上下界|是，包括目标和覆盖它的邻域，基线另有全帧路径|进入所有候选的高斯密度、整行打分、可靠性和学习特征。guard 本身用全活跃网格，非仅载荷。[局部窗口](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/hypothesis_frontend.py#L141-L176)、[启用路径](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/hypothesis_frontend.py#L203-L218)|
|局部对数边际 ℓ、整行发射 E|每个候选的 16 点复高斯，含 1/(πv) 与 1/16；E 为同一数据行 ℓ 的和|本行显式含目标一次；其他 RE 还可能因方差间接含目标|E/τ 进入 HMM；非 HMM 路径按行使用 log prior + E/τ。[边际/整行](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/hypothesis_frontend.py#L243-L257)、[单行权重](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/hypothesis_frontend.py#L286-L289)|
|平滑权重 w|初始先验，stay 对角转移，异状态均分，14 行正反向消息，发射 E/τ|是，目标亦可传播到其他行|B1 取最大，B2a 加权 H，B2b 加权 H 并加候选间方差，B3 将 w 传给原始似然混合器。B0/B0b 固定 LS 分别用基线/局部方差。[HMM](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/hypothesis_frontend.py#L341-L383)、[六个读出](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/group_b_posterior_fusion.py#L93-L144)|
|OFDM B3 目标 LLR|log w 加每个原始目标候选符号对数密度；先对 r 求和，再按比特求和|是|每个目标实际使用的候选权重 ∝ w·exp ℓ。离线真值只用于指标。Dev 选择 τ 和 stay，另拟合各方法读出 LLR 温度；后者改变概率不改变正负。[原始似然混合](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/ofdm_data.py#L350-L383)、[Dev](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/group_b_posterior_fusion.py#L206-L245)、[评估](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/group_b_posterior_fusion.py#L290-L306)|
|PLUTO 先验与方差|8 个当前导频 [13,21)，经验训练先验；历史条目可用其自身载荷，但读取历史切片严格止于当前帧前，邻近和内容先验使用过去条目，内容匹配也看当前导频|对当前载荷目标均否|构造 3 族×4 象限×7 CFO 的 84 候选，r=(4c+q)·7+g；导频估计 h,d,v，温度调整证据与先验。全帧载荷 VV 属另一基线。[当前导频](https://github.com/20171806011/EM-FloatFormer/blob/ade8333fb68126251a83e484b69b821b5d97bcea/research/realdata_pluto_qpsk/receivers.py#L95-L169)、[历史生成](https://github.com/20171806011/EM-FloatFormer/blob/ade8333fb68126251a83e484b69b821b5d97bcea/research/realdata_pluto_qpsk/receivers.py#L194-L222)、[因果切片](https://github.com/20171806011/EM-FloatFormer/blob/ade8333fb68126251a83e484b69b821b5d97bcea/research/realdata_pluto_qpsk/receivers.py#L650-L668)、[邻近](https://github.com/20171806011/EM-FloatFormer/blob/ade8333fb68126251a83e484b69b821b5d97bcea/research/realdata_pluto_qpsk/receivers.py#L728-L738)、[内容](https://github.com/20171806011/EM-FloatFormer/blob/ade8333fb68126251a83e484b69b821b5d97bcea/research/realdata_pluto_qpsk/receivers.py#L353-L405)、[编号、权重和方差](https://github.com/20171806011/EM-FloatFormer/blob/ade8333fb68126251a83e484b69b821b5d97bcea/research/realdata_pluto_qpsk/receivers.py#L435-L484)、[实际调用](https://github.com/20171806011/EM-FloatFormer/blob/ade8333fb68126251a83e484b69b821b5d97bcea/research/realdata_pluto_qpsk/receivers.py#L791-L810)|
|PLUTO B3a|导频/历史 α；目标 y、每分支均值 h·x+d、v_r 与均匀 QPSK|α 否；局部密度是|α 乘局部联合密度，按候选求和、归一化、按 bit 分组；两 bit 各自阈值 >0.5。[B3a](https://github.com/20171806011/EM-FloatFormer/blob/ade8333fb68126251a83e484b69b821b5d97bcea/research/realdata_pluto_qpsk/receivers.py#L536-L559)|
|PLUTO B3b|同一 α，100 个载荷各自的分支边际，条件于候选的目标符号分布|β 是；分支内后验是|β∝α∏m_j，乘当地 **已除以 m_t** 的后验。两种写法同一 bit 后验。B2 的 v 虽计算并用于评估，载荷硬判按均值均衡后最近 QPSK。[B2](https://github.com/20171806011/EM-FloatFormer/blob/ade8333fb68126251a83e484b69b821b5d97bcea/research/realdata_pluto_qpsk/receivers.py#L520-L534)、[B3b](https://github.com/20171806011/EM-FloatFormer/blob/ade8333fb68126251a83e484b69b821b5d97bcea/research/realdata_pluto_qpsk/receivers.py#L562-L580)|

# 生成模型与精确性边界

令 I 为独立于目标的侧信息，r_s 为行状态，x_i 均匀，f_{ri}(x)=CN(y_i;H_{ri}x,v_i)，m_{ri}=Σ_x f_{ri}(x)/16，q_{ri,b}=Σ_{x:b=1}f_{ri}(x)/(16m_{ri})。若参数由 I 固定且噪声条件独立，联合密度为 π(r₀)∏_{s≥1}T(r_s|r_{s−1})∏_i [f_{r_{s(i)},i}(x_i)/16]。定义 c_r 为去掉目标因子后该行的平滑概率，w_r 为含目标的平滑概率。在 τ=1 时 w_r∝c_r m_{rt}。正确目标比特概率既等于 Σ_r w_r q_{rt,b}，也等于 Σ_r c_r m_{rt} q_{rt,b} / Σ_r c_r m_{rt}。现行 B3 的读出却以 w_r 代替后一式的 c_r，造成有效权重 ∝ c_r m²。若各候选 m 相同、候选退化、或分支内 bit 概率相同，输出差可以消失；目标出现两处本身不足以判错。

冻结参数、使用代码的 τ 时，w_r∝c_r m^{1/τ} 仍成立，原因是给定目标行状态，目标发射因子可从所有路径的和中提出。原始 B3 的有效权重 ∝c_r m^{1+1/τ}；显式删除再用原始局部密度则 ∝c_r m；保留含目标权重再混合分支后验则 ∝c_r m^{1/τ}。后两者的等价一般限于 τ=1。τ≠1 的平滑是幂似然，不能直接称为同一物理高斯模型的普通后验。温度校准、由 IQ 拟合的方差、估计参数不确定性和真实噪声相关性都是独立的建模问题。固定提交保存的 Group B Dev 校准为 τ=6.0、p_stay=0.8，B3 的读出温度为 2.3154744812056767。故对应配置冻结其他拟合量后，原读出的局部有效因子为 m^(7/6)，其中可归因于整行发射重复使用的因子是 m^(1/6)。上表 τ=1 的 m² 是验证恒等式的合成条件。Group C 从 Group B 的 JSON 读取 τ 与 p_stay。[校准记录 4–14 行](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/results/group_b_results.json#L4-L14) · [Group C 读取 893–896 行](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/group_c_transport_attention.py#L893-L896)

PLUTO 给定已拟合的 α,H,d,v，取帧共享离散状态 r、独立均匀 QPSK 和条件独立噪声。B3a 等价于只观察目标的联合边际。B3b 满足 β q_{t,b} ∝ α ∏_{j≠t}m_j ·Σ_{x:b=1} f_{rt}(x)/4。固定 d 与事先给定的 σ²、v_h 时，导频复高斯边际的行列式引理和二次型正确。实际还用相同导频估噪、内容匹配后再融合、用 h 的后验经验修正 DC，且载荷预测采用 σ²+var_h+var_d 再缩放；把共享连续 h,d 精确积分一般会在不同载荷间引入相关性。故精确性限于给定插件分支库的离散混合模型。[导频证据与 DC](https://github.com/20171806011/EM-FloatFormer/blob/ade8333fb68126251a83e484b69b821b5d97bcea/research/realdata_pluto_qpsk/receivers.py#L319-L350)、[温度和预测方差](https://github.com/20171806011/EM-FloatFormer/blob/ade8333fb68126251a83e484b69b821b5d97bcea/research/realdata_pluto_qpsk/receivers.py#L461-L472)

# 学习头的输入和解释

|量|实际输入与目标路径|如何进入输出|
|---|---|---|
|4 个复特征|可靠性加权的均衡软反馈相关、软残差、四次幂都用目标所在行的载荷；第四个候选相位由状态索引预定。权重依赖网格方差。|复通道拆为实、虚；物理网络使用跨行、跨候选的信息，相关相量还直接进入连续相位读出。[复特征](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/hypothesis_frontend.py#L259-L292)、[输运注意力](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/group_c_transport_attention.py#L293-L384)、[物理网络](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/group_c_transport_attention.py#L419-L459)|
|10 个标量|行内居中打分、跨帧内各行均值的候选打分、先验、软置信度、软残差 MSE、时间、斜率、cos 跳相、sin 跳相、超额能量与 log-SNR 合项；第 1、2、4、5、10 项可受目标影响。第 2 项在所有行广播。|物理网络与两个匹配的实网络共用；实 Transformer 做行和候选注意力，MLP 拼时间均值与候选均值。[标量次序](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/hypothesis_frontend.py#L294-L324)、[实 Transformer](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/group_c_transport_attention.py#L503-L537)、[MLP](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/group_c_transport_attention.py#L562-L581)|
|基准权重、读出和标签|缓存用 **另算的 HMM 平滑权重**，不使用 bundle 的逐行 base_logw_sr。网络前向只取复实虚、标量、基准权重。另存 y、h_ls、h_ca、nv 供解调；标签存作监督/评估。|三个头输出校正权重、每行每候选相位、每行方差缩放；detach 只切断梯度，数值仍依赖权重。解调乘真实目标 y。省略的目标观测模方 和 log(πv) 在同一 RE 各候选共用 v 时抵消。[缓存](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/group_c_transport_attention.py#L666-L686)、[实际前向](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/group_c_transport_attention.py#L708-L723)、[读出](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/group_c_transport_attention.py#L156-L202)|

学习模型可作为条件判别模型接收目标 IQ。相同观测进入特征和读出并不推出标签泄漏，也不能自动把学到的门控解释为上述生成模型的候选先验或精确后验。仅从物理头的重复指数不能推算 C 的误差。要识别 C 对目标的额外依赖，必须以同一冻结权重比较三次前向，明确哪些残差和校准被冻结。

# 两种删除如何解释

- **显式删除**　保留原始已拟合 H、v、其他 RE 的分数及非 HMM 特征。在目标行对每个候选减去 ℓ_{rt}，再走原 HMM 与原始 16-QAM 读出。等价于归一化 log w−ℓ/τ。这里只测显式权重路径。可选学习模型接口冻结原前向的门控残差、相位和方差校正，单独改基准门控。

- **排除重建**　在目标所在行重算有效计数。全活跃功率上限去目标；若 s=2，重算截尾集合与截尾数量。对 3×3 窗口同时去掉该样本的能量和 edge 复制的每一次计数，重算基线、局部方差、全网格剩余打分和 HMM。用新方差、新掩码重算软置信、残差、相量、整帧相对打分和能量特征。在行超额能量均值中也排除目标中心，这是本探针明确选定的缺失处理约定。预测最后才读原目标 y，使用重建的 v_t。此探针同时改变多条途径，不能把与显式删除之间的全部差值单独归因给 NBI 或学习特征。

- **冻结回放**　原始模式调用固定提交；原代码无修改。改目标 IQ 时，排除重建的所有拟合参数、分数、特征和 HMM 权重应完全不动，但目标分支内后验与最终比特通常会动。各目标必须单独重建，不能用一张同时删多点的网格冒充逐点结果。输入 NPZ 表示一个已选条件的 TTI；train-only 参考和 Dev 校准参数由回放者固定传入。

# 已运行的独立最小枚举

`mini.js` 以 2 行 × 3 子载波、2 候选和完整 16-QAM 枚举所有四条状态路径。基线方差固定 0.12；NBI 保留 3×3 边界复制、阈值、去目标重算分母；行均能是用于验证排除操作的诊断量。该缩小网格只验证两种运算，不模拟训练参考库或 17 状态性能。目标 (0,1)，y=−0.88+2.63i，是从合成输入中选来展示判决翻转的例子。τ=1，表中 bit 为自然二进制顺序。

|模式|ℓ(r=0,1)|目标方差|HMM 权重 (0,1)|第 0 位分支内 P(1) (0,1)|四位最终 P(1)|硬判|
|---|---|---:|---|---|---|---|
|原读出|−5.80352, −6.33711|1.07676|0.45526, 0.54474|0.15103, 0.86262|0.44447, 0.49896, 0.99240, 0.89573|0011|
|仅删显式项|同上|1.07676|0.32893, 0.67107|同上|0.53866, 0.53552, 0.99126, 0.89021|1111|
|排除重建|−25.32224, −29.58050|0.12|0.28641, 0.71359|0.00000642, 0.99999935|0.03405, 0.10003, 1.00000, 1.00000|0011|

独立 QPSK 例含两个分支和两个载荷，目标为载荷 0（映射帧坐标 26）。B3a 两位概率为 (0.283321, 0.989415)；B3b 的 β=(0.737880,0.262120)，两位概率为 (0.252194,0.990046)，硬判 01。全枚举 2×4×4 个候选/符号组合与 cavity 写法均通过 1e−12 容差。`mini.json` 包括全部候选、每个分支的每一位、使用的门权和解调后的有效权重，另含 τ=2 对照和扰动目标后重建不变的断言。

# 独立的全尺寸合成网格校验

`fullgrid.js` 对一帧 14×128 合成 IQ、101 条活跃载波和全部 17 个候选，按固定提交公式独立计算双向 HMM、完整 4 个复特征和 10 个标量特征。这一组刻意使用平坦已知导频/参考，故 LS 与内容寻址都为 1，导频二阶差分为零。三处目标逐一扰动，重建后的全网格方差、所有候选分数和全部特征均严格不变。在 τ=1 时，两种等价物理写法通过 1e−10 容差。

|目标 (s,活跃索引; FFT)|原始→重建的本地 v|第一位 P(1)，原始 / 显式删除 / 重建|重建后标量特征最大绝对变化|
|---|---:|---:|---:|
|7,45; 58|1.681706 → 0.016286|0.786868 / 0.786868 / 0.790755|0.211853|
|2,39; 52|0.016286 → 0.016324|3.85e−37 / 3.85e−37 / 4.68e−37|0.013880|
|13,100; 114|0.016286 → 0.016286|0.251809 / 0.251809 / 0.251809|0.039641|

平坦导频行各候选 H 相同，显式删除在那里不能改变比特。第三个目标中物理读出近似相同，但用于学习头的特征已变。`fullgrid.json` 记录候选 0–16 的局部边际、四位分支内概率、权重、最终概率和硬判。以上为独立公式实现的结果。

# 回放

```sh
node mini.js > mini-replay.json
node fullgrid.js > fullgrid-replay.json
python audit.py --ofdm /path/to/ofdm/research/realdata_ofdm_stress_and_theory --pluto /path/to/pluto/research/realdata_pluto_qpsk > synthetic.json
python audit.py --ofdm /path/to/ofdm/research/realdata_ofdm_stress_and_theory --frame one-stressed-tti.npz --target 7 45 --tau 2 --stay .88 > one-target.json
```

`audit.py` 校验两提交对应文件的 git blob SHA。固定模型参数时 `physical(...,model=cpu_model.eval())` 也输出学习头各模式的分支/最终概率；`pluto(y, bank, target, receivers)` 接收原始 126 符号和对应分支库。OFDM 的 llr、prob 都是 Dev 最终 LLR 温标之前的值。若复核校准概率，另算 sigmoid(llr/T_cal)。默认探针检查普通行、导频行数据位、边界 RE；PLUTO 单帧用合成导频与三组指定先验。没有训练。本文记录 B 组会合前的单目标检查。会合后的实验状态、归档 240 帧复算与实测输入可用性见[总报告](../README.md)。
