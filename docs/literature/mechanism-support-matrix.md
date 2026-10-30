# 机制支撑矩阵（第二轮）

三个技术问题并列开放，不预设贡献是"接收方门控"。每个问题下按"现有办法 / 学习信号 / 理论前提 / 缺口 / 可复用组件 / 证据等级"组织。

**证据等级**：
- `正-全文` = 本轮读了正文相应章节
- `正-摘要` = 只核实了摘要或会议页
- `题录` = 只有元数据
- `未取得` = 知道存在但拿不到

---

## A. 失败识别与归纳：可观测的不良结果如何关联到"条件—操作—后果"

### A0 两个必须先区分的前提

1. **termination ≠ truncation**。Gymnasium 官方文档与 Pardo et al. (1712.00378) 明确：termination 是 MDP 内的终止（含成功、失败、跌倒），truncation 是时间/边界截断，**不是失败**；且二者在 bootstrap 上处理不同。任何用"终止=失败"的实现在超时环境里会把超时误当失败。
2. **两个"premature termination"不是一回事**。MAST (2503.13657) 的 failure mode "Premature termination" 指 LLM agent 提前停止工作，不是环境终止。

### A1 现有办法

| 办法 | 代表工作 | 失败被定义成什么 | 需要的监督 | 在线可得 | 证据等级 |
| --- | --- | --- | --- | --- | --- |
| 环境终止 | FEMA (2603.07110) §3.2 | 训练 rollout 意外终止 | 无额外监督 | 是 | 正-全文 |
| 约束违反 / 外生代价 | MICE (2601.11953, PMLR v267:18290–18316)；CPO (1705.10528) | 外生代价 > 0 的状态 / 约束违反 | 需要代价通道 + 设计者定阈值 | 是（若有代价） | 正-全文 |
| 任务结果（attempt 级） | FAR (2607.01111 v2 2026-10-07) §4.1 | 任务未完成 / 超时 | 任务成功标志 + 预训练 critic 的 ΔV | attempt 之后（事后） | 正-全文 |
| 低回报 | FEMA；CBR (2110.08470) | FEMA：失败段 Monte-Carlo return；CBR：只 retain 正奖励 | 回报阈值 | 是 | 正-全文 |
| 事后重标注 | HER (1707.01495) §3.3 | **不定义失败**：失败轨迹换 goal 后成为成功，不产生负例 | 二值 goal 奖励 | 是 | 正-全文 |
| TD 误差 / 优势 | PER (1511.05952)；SUPER (2311.00865) | **不是失败**：学习优先级 / 可学性代理 | 无 | 是 | 正-全文（SUPER） |
| 灾难性结果 + 规则 | Shperberg et al. (PMLR v199:231–242)；ShieldPPO (2202.09516) | 可识别的 catastrophic (s,a) | Shperberg 需**人类/预言机 MAE**提供谓词规则 | 结果检测在线；规则归纳**非自动** | 正-全文 |
| 无失败数据的运行时预测 | FAIL-Detect (2503.08558, RSS 2025)；FIPER (2510.09459, NeurIPS 2025) | 策略失败/即将失败（OOD + 动作不确定性 + conformal） | 只需成功数据 | 是 | 正-摘要 |
| 成功-失败数据训练的风险图 | RoboMD (2412.02818) | 失败区域 vs 成功区域 | 有限 success-failure 数据 | 离线训练、在线查询 | 正-摘要 |
| 学习式安全 critic | Safety Critic (2010.14603) | 安全事件（跌倒、掉落） | cost 信号 | 是 | 正-摘要 |
| 回报分布 | QR-DQN (1710.10044) | **不定义失败**，只建模回报分布 | 无 | 是 | 正-摘要 |

### A2 不能当失败标签的信号

- **TD 误差 / 优势**：高 TD 误差也可能是"好意外"或探索。PER 用作优先级，SUPER 用作"可学性"代理且作者自称 "merely a heuristic"。**文献中没有把它定义为失败并验证的工作**（本次检索未发现）。
- **终止本身**：成功与失败都会终止；不区分 truncation 会误判超时。
- **低回报**：受 horizon、探索、reward scale 影响；不归一化无法跨任务比较。
- **OOD / 不确定性**：benign OOD 与真失败混淆；FIPER 用双指标 + conformal 压误报，但仍不输出失败类型。

### A3 因果归因与反事实的真实状态

| 做法 | 代表工作 | 归因对象 | 是否需要额外监督 |
| --- | --- | --- | --- |
| 反事实信用分配 | Mesnard (2011.09464)、COCOA (2306.16803)、Shapley CF (2106.00285) | **奖励**对动作/agent 的贡献 | 无（COCOA 需模型） |
| 多智能体失败归因 | Who&When (2505.00212, ICML 2025) | 失败的**责任 agent / step** | 需要人工细粒度标注；自动版用 LLM |
| 动作块级失败归因 | FAR (2607.01111) | 失败 attempt 中 ΔV 最低的 chunk | 任务结果 + 预训练 critic |
| 根因 (s,a) + 专家规则 | Shperberg (PMLR v199) | catastrophic outcome 的 (state, action) 等价类 | 人类/预言机 MAE |

- **关键判据**：Who&When 报告 "53.5% accuracy in identifying failure-responsible agents but only 14.2% in pinpointing failure steps"（摘要）。这说明**自动把失败归到具体步骤仍然很难**，任何声称"定位失败操作"的方法都需要给出与这个难度相称的证据。
- 反事实信用分配解决的是奖励归因，**不告诉你"这一步导致了失败"**。二者不能互相替代。

### A4 "归纳"层面：失败能否变成可复用结构

| 归纳形式 | 代表工作 | 存正/负例 | 监督来源 | 备注 |
| --- | --- | --- | --- | --- |
| state-action 谓词规则 | Shperberg (PMLR v199) | 负例 | **人类/预言机 MAE** | "reliance on an external MAE is an obvious limitation"；自动 MAE 留给未来工作 |
| 自然语言 mistake notes | MNL (2512.11485) §3.3 | 负例 | 任务 ground truth 或 LLM judge | 五段结构：Corrected Examples / Correct Approach / Mistake Summary / Generalizable Strategy / Anti-Patterns |
| 失败模式分类 | MAST (2503.13657) | 负例 | 6 名专家标注 150 条 trace + LLM judge | 14 个 mode、3 类；per-trajectory 标签，不是 condition-action 规则 |
| 案例库 | Atzeni et al. CBR (2110.08470) §3 Retain | **只存正例**（"whenever r_t > 0"） | reward > 0 | 相似度 = 共享 VQ code 的比例；**复用时不更新参数** |
| 失败轨迹 → 偏好对 | ETO (2403.02502) | 负例 | 任务成功/失败 | 用 DPO **更新参数**，非部署期 |
| 失败 → 动作条件后果模型 | FACT (2608.10232) | 负例（success=0 rollout） | 任务成功/失败 | 学连续 future video + progress，不是符号三元组 |
| 负示范 | UNIQ (2410.08307) | 负例 | 需预先给定的不良示范数据集 | 离线 |
| 策略规则抽取 | VIPER (1805.08328) | 不涉及失败 | 已训练策略 + Q | 蒸馏的是策略，不是失败 |

### A5 缺口（检索性表述）

- **未发现**只从一次失败自动归纳出**符号化**"条件—动作—后果"三元组、并跨"相关但不完全相同"情境复用的工作。最接近的三条各缺一角：Shperberg 有符号结构但规则靠人给；MNL 有可复用笔记但是自然语言；FACT 有动作条件后果但是连续模型。
- **未发现**把 TD 误差/优势定义为失败并做验证的 RL 工作。
- **未发现**对"意外"与"失败"给出形式化区分和可操作判别器的工作。
- **未发现**存储负例的 CBR-RL 工作；Atzeni 只存正例。
- **未发现** FTA/FMEA 与 RL 自动归纳结合的开放全文（Grimmeisen 2024 付费墙）。

### A6 可原样复用 / 需改

| 组件 | 状态 |
| --- | --- |
| termination/truncation 分离 | **必须先做**，否则失败标签错误 |
| 终止 + 最后 K 步窗口（FEMA） | 可原样复用 |
| 核密度伪计数（MICE / NGU 的 ξ） | 可原样复用 |
| ΔV 分位判定失败块（FAR） | 可复用，但依赖预训练 critic |
| 谓词规则 + 屏蔽（Shperberg） | 可复用结构，但规则来源需自建 |
| mistake notes 五段结构（MNL） | 可复用为"归纳产物"的字段设计 |
| failure-mode 词表（MAST） | 可复用为评估分类，但需人工标注 |
| 符号化"条件—操作—后果"自动归纳 | **需自建**，无现成实现 |

---

## B. 跨 agent 适用性：发送方的严重失败未必对接收方有价值

> **第二轮结论修正**："逐接收方判断"本身**不是空白**——相似度、接收方不确定性、接收方历史结果、注意力/队友模型都有先例。空白在更窄的交叉：**以发送方失败为对象、用接收方自身后续结果校准、在执行期使用**。这三条同时满足的工作，本次检索**未发现**。

### B1 适用性信号的五级谱（从最便宜到最接收方特定）

| 级别 | 信号 | 代表工作 | 用到的接收方信息 | 阶段 | 证据 |
| --- | --- | --- | --- | --- | --- |
| 1 | 发送方 TD 误差 | SUPER；INSPIRE 的一级评分 | **无** | 训练 | 正文 |
| 2 | 成对相似度 | Hayashida 2025（观测 L2 + 策略 TV 距离 + 未学习度熵）；CMU-RI-TR-25-11（策略相似度 + CE 阈值） | 双方观测/策略 | 训练 | 正文 |
| 3 | 接收方自己的 TD 误差 | INSPIRE 的二级过滤（"evaluates each experience using the local value function Q^i to compute its TD error", §3.4） | 接收方 Q 网络 | 训练 | 正文 |
| 4 | 接收方不确定性下降 | T2MAC (AAAI-24)：通信价值 = 通信前后接收方不确定性差 | 接收方 belief | 训练 selector / 执行使用 | 正文 |
| 5 | 接收方历史结果 | XferAgent (arXiv 2610.19581)：cross-agent outcome score，且**排除自身结果**（leave-one-agent-out） | 接收方任务 + 其他 agent 的过往结果 | **执行期** | 正文 |

- 1–2 级最便宜但最不接收方特定；4–5 级才真正使用接收方反馈。
- **没有任何一级把"发送方失败的严重程度"当作负信号**：现有 sender TD error 全部被当作"有用性"的**正**代理。

### B2 逐接收方生成的机制（不等于适用性判断）

| 工作 | 机制 | 是否判断"该经验对我是否适用" |
| --- | --- | --- |
| TarMAC (ICML 2019) | sender signature × receiver query 注意力 | 否，是当前消息相关性 |
| MAIC (**AAAI-22**，非 2021) | 逐队友模型 z_ij + 互信息正则 + 稀疏权重 | 否，是协调收益 |
| PMAC (AAAI-24) | 学习式个性化收发，p(m_ij \| h_i, h_j) | 否，是架构设计 |
| CACOM (AAMAS 2024) | 依 helpee context message c_i 生成个性化消息 | 否，是端到端发送决策 |
| Agent KB (2025) | disagreement gate：原 plan 与 refined plan 的余弦 ≥ 0.8 | 否，比较的是两个计划，不涉及接收方身份 |

### B3 负迁移的证据状态

| 证据 | 来源 | 说明 |
| --- | --- | --- |
| share-all 可低于 no-sharing | SUPER §5.6/Table 1（Pursuit 181.43 vs 148.27） | 直接证据，但针对转移共享，非失败消息 |
| 固定比例共享会造成暂时退化 | Hayashida 2025 §4.2，10 trials | 直接证据；抑制不相似共享可避免恶化 |
| 知识跨域不对称 | Agent KB §4.3："SWE knowledge does not generalize to reasoning tasks" | 直接证据，文本经验 |
| 非专家教师需要拒绝机制 | EAA (**ICRA 2023**，非 AAMAS 2022) §IV-C | 间接证据 |
| 非对称域下发送方 TD 会失效 | SUPER §6 作者自述 "merely a heuristic ... we do not expect to work in entirely asymmetric domains" | **预期，非实测** |

**关键**：**没有任何一篇测量"失败消息"对接收方的负迁移**。因此"发送方严重失败未必对接收方有价值"目前是**合理假设**，不是已被文献直接验证的结论。

### B4 INSPIRE 的准确机制（正文已取得，修正第一轮）

- Zhang, Xie, Yang, Jin, Song, *Machine Learning* 115:95, 2026。
- **两级**：(1) Neighbor Experience Transmitter 用邻居 TD 误差的高斯分布统计**聚合评分**发送方经验；(2) 接收方再用**自己的 Q^i** 算 TD 误差做二次过滤。
- 所谓 "neighbor feedback" 是**邻居 TD 误差的分布统计**，不是接收方对经验价值的显式反馈。
- 论文自述 "operates under a homogeneous MARL setting"，**完全异构是 future work**。
- 阶段：训练期。

### B5 缺口（检索性表述）

- **未发现**同时满足这五条的工作：以发送方严重失败为对象 / 逐接收方判断 / 用接收方后续结果校准 / 执行期无梯度 / 异构或非对称角色并报告逐接收方负迁移。
- **未发现**把"发送方失败严重程度"作为适用性**负信号**的机制。
- **未发现** failure-specific 的 accept/reject 分类器，或对失败经验适用性做校准的指标。
- **未发现**跨 agent 失败经验迁移的 held-out related-situation benchmark（现有评测多是训练期样本效率或团队平均回报）。
- **未发现**在非对称/异构设定下专门测量 failure message 负迁移的工作。

### B6 可复用 / 需自建

| 组件 | 状态 |
| --- | --- |
| 发送方 TD 误差筛选（SUPER） | 可原样复用，**必须作基线** |
| 邻居 TD 分布统计 + 接收方本地 TD 二次过滤（INSPIRE） | 可复用，是当前最强启发式基线 |
| 成对相似度（Hayashida：L2 + TV + 熵） | 可复用，10 trials 的口径可直接沿用 |
| 接收方不确定性下降（T2MAC） | 可复用于"消息价值"的度量 |
| 接收方历史结果（XferAgent 的 LOAO） | 可复用，且是唯一执行期、用接收方结果校准的先例 |
| 以接收方失败率为目标的适用性学习 | **需自建** |
| 失败严重程度作为负信号 | **需自建**，无先例 |

### B7 最小可辩护实验设计（来自证据的约束，不是拍脑袋）

- 任务：非对称角色 + held-out related states。
- 对照至少包括：no sharing / state-only / shuffled-mismatched failure / stale failure / sender-TD 选择 / 成对相似度选择 / receiver-own-TD 过滤 / receiver-uncertainty 选择 / receiver-outcome-history 选择。
- 指标：**逐接收方**的重复失败率与对照状态成功率，必须报告负迁移，不能只看团队均值或发送方回报。
- seed：Hayashida 用 10、CMU 用 5；建议至少 10。



---

## C. 任务驱动的高效传输：传什么、给谁、何时、多长

> **第二轮结论修正**：文献里确实有把"失真"定义为**任务价值损失**的工作（Mostaani 的 rate-distortion；RAVEN 的接收方条件化 action-value 码；"Emergence, Not Bandwidth" 的 receiver-advantage 统计量）。第一轮说"没有任何工作报告时延/语义对齐"是错的——错在把范围限定在 MARL 经验共享，而正确结论是：**在"失败经验传输"这一具体对象上未见**，在通信学习这个大类里已有。

### C1 四个量纲必须分开

| 量纲 | 单位 | 本轮确认报告了该量纲的工作 |
| --- | --- | --- |
| 消息大小 | bit / byte per message | RAVEN（2 bit/message）、IB-VQ（4 bit、800±85 bit/episode）、CGIBNet（5 bit/message, Traffic Control）、ETCNet（170/100/60 bit/s）、DSMS（B=64，字节级）、Mostaani（2 bit/channel use）、Emergence（2.000 bit/decision） |
| 通信频次 | message/episode、发送概率 | ETCNet（实测 46%/24%/14%）、SUPER（1%–10% 带宽）、MAIC（后验剪枝 0–90%）、Message Utility Estimation（尝试数 −42.5%） |
| 编码/推理时延 | ms / s / FLOPs | Mem0（search p50 0.148s、p95 0.200s）、2601.07978（200ms 档、4–5% 惩罚）、RAVEN（FLOPs −82.2~96.8%） |
| 样本/墙钟 | env steps / 小时 / 美元 | SCoUT（4M steps、20 seeds）、RAVEN（1.2M/2.05M steps）、Agent KB（$0.27 vs $86.0） |

**警告**：`bandwidth` 一词在文献里被用来指代上述至少三种不同东西。引用时必须说明是哪一种。

### C2 把"语义"对齐到任务价值的三个先例

| 工作 | 失真/价值如何定义 | 关键短摘录 |
| --- | --- | --- |
| Mostaani et al. (2005.14220) | distortion = \|V*(s) − V*(h(s))\|，**任务价值损失**，不是重建误差；约束 log2\|C\| ≤ R，R=2 bit/channel use | "minimize \|V*(s(1))−V*(h_i(s(1)))\| subject to log2\|C\|≤R" |
| RAVEN (2609.37566) | 在**接收方自身 context** 内最小化 centered action-value profile 的失真，得到 4 符号（2 bit）码 | "Each message costs log2 K=2 bits" |
| Emergence, Not Bandwidth (2609.34373) | 理论：最优消息是**接收方 advantage 的统计量**，不是发送方观测的统计量；固定 2.000 bit/decision | "a statistic of the receiver's advantage, not of the sender's observation" |

这三条合起来给出一个可用的判断：**"传什么"的最优答案取决于接收方的决策价值，而不是发送方的状态**。这正好支持"用接收方损失定义适用性"的路线，但也意味着这条路线在通信学习里**已经被理论化**，不能当作全新想法。

### C3 何时传（触发）

| 机制 | 工作 | 触发信号 | 是否有显式预算 |
| --- | --- | --- | --- |
| 二值发送门 | IC3Net (1812.09755) | 隐状态 → 2 动作 softmax | 无 |
| 事件触发 + 零阶保持 | ETCNet (2010.04978, TNNLS 2021) | 带发送概率上限的 constrained MDP，Lagrangian | 有：N(N−1)LFp ≤ n |
| 效用阈值 | Message Utility Estimation (2304.06873) | U ≥ T | 无 bits |
| Shapley 值查询 | SMV (IJCAI-22) | 只查询 SMV > 0 的队友 | 无 bits |
| 接收方不确定性下降 | T2MAC (AAAI-24) | 通信前后接收方不确定性差 | 无 bits |
| 接收方效用 Q 值 | SCoUT (RLC 2026, 2603.04833) | pairwise utility Q；每 K=10 步 | 无 bits，budget 列为 future work |

- **ETCNet 是唯一同时给出"何时"与"多少 bit/s"的模板**，可直接改造。
- 其余多为"何时"，不控"多少"。

### C4 传多长（预算分配）

- **DSMS**（TMC 2024, 23(12):15080–15097；arXiv 2306.10134）给出 per-agent 预算分配：b_i = 2·soft_ceil((B/2−n)w_i)，Σb_i ≤ B，最小 2 bytes。
- **IB-VQ**（2602.02035）给到 bit/episode 级：4 bit/message，800±85 bit/episode vs 2800±180 full；但论文内部 71.4% 与 41.4% 两个降幅不一致，复现需锁定口径。
- **CGIBNet**（仅 arXiv 2112.10374，**无期刊版**）用相对 SCR/MCR 百分比，Traffic Control 每消息 5 bit；无绝对 bytes/step。

### C5 检索与压缩的成本

- MARL 中**未发现**报告 kNN/向量库检索时延或索引存储的工作。
- LLM-agent 侧有可借的记账协议：Mem0 报告 search p50 0.148s / p95 0.200s；2601.07978 报告 200ms 延迟档、8 Mbit/s、50ms jitter 与 CPU/RAM/disk/network 成本；Agent KB 报告检索成本 $0.27 vs 总评测 $86.0。
- **这些数字不能迁移到 MARL**，但**协议可以**：p50/p95 时延、索引内存、每查询 byte/token。

### C6 缺口（检索性表述）

- **未发现**以"失败经验"为消息内容、同时报告 bit/byte 预算的工作。
- **未发现** MARL 执行期检索的时延/存储成本报告。
- **未发现**"等语义信息量"的受控对照：现有工作要么等名义带宽（ETCNet 目标 50% 实测 46%），要么等消息数，没有等"决策相关信息量"的。
- SCoUT 已核实**真实存在**（RLC 2026 + arXiv 2603.04833），但**无 bit 预算**，budget 列为 future work。

### C7 可原样复用 / 需改造

| 组件 | 状态 |
| --- | --- |
| ETCNet 的 constrained-MDP + ZOH | 可原样复用结构，事件定义需改为失败检测 |
| DSMS 的 per-agent 预算分配公式 | 可直接用，Fourier resize 换成字段/token 预算 |
| RAVEN 的 receiver-conditioned 码本 | 可复用，前提是离散动作 + 可集中训练 |
| Mostaani 的 task-value distortion | 可直接作为训练正则与"语义对齐"度量口径 |
| SMV 的 Shapley 消息价值 | 可作离线标签，N 大时不可行 |
| Mem0 / 2601.07978 的成本记账协议 | 协议可复用，数值不可迁移 |
| IC3Net / CGIBNet / DVQ / MAIB 式门控与压缩 | **需改造**：它们是逐 step、同质、无事件语义 |
| SCoUT 式 utility 分组 | **需改造**：K=10 分组与失败事件不对齐，且无 bit 预算 |

### C8 评测协议（来自证据的约束）

1. 四个量纲分开报：byte/message、message/episode、encode+transmit latency、env steps/wall-clock。
2. **等比特必须等实测消耗**：ETCNet 目标 50% 实测 46%，说明"目标"不等于"实测"。
3. 语义对齐用**任务价值失真**（Mostaani/RAVEN），不要用重建 MSE 或单纯 entropy。
4. 至少两类语义对照：同比特不同内容（随机/打乱消息）、同内容不同比特（压缩/不压缩）。
5. 经验共享的接收方相关性，应直接测**接收方重复失败率或接收方 value loss**，而不是发送方 TD 误差。


---

## 三个问题的关系

- A 决定"经验里有什么"，B 决定"给谁"，C 决定"怎么给"。
- 它们**不互为前提**：可以只做 A（例如提出一种无标注的失败分解），也可以只做 B（例如用后果等价性替代状态相似度），也可以只做 C（例如给出每比特收益的评测协议）。
- 但任何一条单独成立，都需要一个能同时容纳另外两条的实验设置，否则无法测。
