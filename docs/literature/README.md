# 文献调研总览

截止日期：2026-10-29（第二轮）。性质：**广筛 + 定向正文核实 + 有限引用追踪**，不是系统综述，不声称穷尽。
**第一轮的强结论已在 [`revision-notes.md`](revision-notes.md) 中逐条更正，请以该文件为准。**

## 证据强度声明（请先读这一节）

第一轮把题录级与正文级证据混用，并据此推出存在性结论。本轮改正为四档：

| 档 | 含义 | 本目录中的写法 |
| --- | --- | --- |
| 档 1 | 读了正文相应章节，该文**确有**此实现 | 附章节/公式/短摘录 |
| 档 2 | 读了正文，该文**未描述** | 明确写"未描述"，不写"没有" |
| 档 3 | 本次检索**未发现** | 明确写检索范围与截止日 |
| 档 4 | **未取得全文** | 不作技术判断 |

当前体量：CSV 收录 **66 条**；**正文级阅读约 20 篇**（含子 Agent 核对：SEAC、SUPER、Pesce & Montana、LeGEM、CooT、FEMA、MICE、Agent KB、MemCo、Atzeni-CBR、Shperberg、INSPIRE、Hayashida、CMU-RI-TR-25-11、MAIC、TarMAC、EAA、T2MAC、RAVEN、Mostaani、ETCNet、DSMS）；摘要级 15 篇左右，其余为题录。**"正文级"不等于精读，多篇只覆盖前 100k 字符或特定章节。**

## 结论（按问题组织，不按论文）

### 问题 A：失败如何被识别与归纳

- "失败"至少有五种互不等价的可观测入口：环境终止（FEMA）、约束违反（MICE/CPO）、任务结果（FAR）、低回报（FEMA/CBR）、人类/预言机规则（Shperberg）。TD 误差与优势**不是失败标签**（SUPER 作者自称 "merely a heuristic"；PER 用作优先级）。
- 必须先区分 **termination 与 truncation**，否则超时会被误判为失败。
- 归纳层面：Shperberg 有符号规则但**规则由人给**；MNL 有可复用笔记但**是自然语言**；FACT 有动作条件后果但**是连续模型**。三者各缺一角，**未发现**自动产出符号化"条件—操作—后果"的工作。
- 失败归因很难有硬数据：Who&When 报告中 agent 级 53.5%、**step 级仅 14.2%**。

### 问题 B：跨 agent 适用性

- **第二轮更正**："逐接收方判断"不是空白。现有信号有五级：发送方 TD 误差 → 成对相似度 → 接收方自己的 TD 误差 → 接收方不确定性下降 → 接收方历史结果。
- 最接近的两条：**INSPIRE**（邻居 TD 分布统计 + 接收方本地 Q^i 二次过滤，训练期、同构）、**XferAgent**（执行期、用其他 agent 的历史结果且排除自身，LLM 路线）。
- **未发现**同时满足「以发送方失败为对象 / 用接收方后续结果校准 / 执行期无梯度 / 异构并报告逐接收方负迁移」的工作；也**未发现**把失败严重程度当**负信号**的机制。
- 负迁移只有类比证据（SUPER share-all、Hayashida 固定比例退化、Agent KB 跨域不对称），**没有一篇测量"失败消息"的负迁移**。

### 问题 C：高效传输

- 四个量纲（bit、频次、时延、样本/墙钟）必须分开报；`bandwidth` 一词在文献里至少指其中三种。
- **第二轮更正**："把失真定义为任务价值损失"已有先例：Mostaani 的 value-loss distortion、RAVEN 的接收方条件化 action-value 码（2 bit/message）、"Emergence, Not Bandwidth" 的理论（最优消息是**接收方 advantage** 的统计量，2.000 bit/decision）。
- 缺口不在方法而在对象：**未发现**以"失败经验"为消息内容、同时报告 bit 预算的工作。
- 详见 [`mechanism-support-matrix.md`](mechanism-support-matrix.md) C 节。

## 最危险的近邻（第二轮排序）

| 排名 | 工作 | 重合的部分 | 不重合的部分 |
| --- | --- | --- | --- |
| 1 | **MemCo**（arXiv 2610.07376, 2026） | 跨 agent 共享 + 显式失败记录 + 按接收方状态/决策阶段条件化检索 + 执行期 | LLM 文本路线；适用性是启发式打分 |
| 2 | **INSPIRE**（Mach Learn 115:95, 2026） | 接收方自己的 Q^i 参与经验筛选——"接收方条件化适用性"不再是空白 | 训练期转移共享；同构；无失败语义 |
| 3 | **CooT**（ICML 2026） | 多智能体 + 跨局上下文 + 执行期无梯度 | 学习信号是专家最优动作监督，不是自身失败后果 |
| 4 | **FEMA**（2026） | 失败→记忆→检索→改变动作的闭环 | 单智能体；手工相似度 + 阈值 |

> RAVEN（2 bit 接收方相关消息）在问题 C 上是必须对照的上界，但对象是当前协作价值而非失败经验。

## 文件索引

| 文件 | 内容 |
| --- | --- |
| [`revision-notes.md`](revision-notes.md) | **必读**：14 条更正与依据 |
| [`papers.csv`](papers.csv) | 筛选矩阵（30 列） |
| [`search-log.md`](search-log.md) | 两轮检索留痕、引用追踪、已知缺口 |
| [`nearest-neighbor-matrix.md`](nearest-neighbor-matrix.md) | 六族对照 + 四档证据分类 |
| [`mechanism-support-matrix.md`](mechanism-support-matrix.md) | A/B/C 三问题的机制、学习信号、理论前提、缺口、可复用组件 |
| [`candidate-method-briefs.md`](candidate-method-briefs.md) | 3 份候选草案（非已批准架构） |
| [`research-opportunities.md`](research-opportunities.md) | 第一轮切口（结论已按 revision-notes 弱化） |
| [`cards/`](cards/) | 核心卡片；正文级 9 篇，其余摘要级 |
| [`../base-stack-candidates.md`](../base-stack-candidates.md) | 分层候选栈 |

## 卡片阅读分级

| 卡片 | 论文 | 深度 |
| --- | --- | --- |
| P01 | Tan 1993 | 摘要 |
| P02 | SEAC 2020 | 正文 |
| P03 | SUPER 2023 | 正文（第二轮更正） |
| P04 | Pesce & Montana 2020 | 正文（第二轮补充梯度核实） |
| P05 | LeGEM 2022 | 正文 |
| P06 | CooT 2026 | 正文（第二轮重写） |
| P07 | FEMA 2026 | 正文（第二轮重写） |
| P08 | MICE 2025 | 正文（第二轮重写） |
| P09 | Agent KB 2025 | 正文（第二轮重写） |
| P10 | ExpeL 2024 | 摘要 |
| P11 | CMU-RI-TR-25-11 | 摘要（PDF 未解析） |
| P12 | What and When to Share 2022 | 摘要 |
| P13 | MemCo 2026 | 正文 |
| P14 | Shperberg 2022（Rule-based Shield） | 正文（子 Agent 核对） |
| P15 | INSPIRE 2026 | 正文（子 Agent 核对） |
| P16 | RAVEN 2026 | 正文（子 Agent 核对） |

## 本目录不包含

不含算法选择、不含任务选择、不含新颖性结论。候选草案是待检验的假设，不是已证创新。

## 已知缺口

1. 前后向引用扩展只对 SUPER、CooT、Pesce & Montana 三篇做了，且只取到题录。
2. 时延、比特级成本、语义信息量控制三处，文献与实验都缺。
3. 付费墙工作（CESDQL、DECAF、INSPIRE 正文）未取得。
4. 中文文献仅 1 篇日文期刊。
