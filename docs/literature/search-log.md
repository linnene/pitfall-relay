# 检索日志

执行日期：2026-10-29（本地时区当日）。执行者：文献调研 Agent。
检索截止：2026-10-29。**不声称穷尽**；2026 年 10 月之后的在线更新未纳入。

## 平台与工具

| 平台 | 用途 | 限制 |
| --- | --- | --- |
| Web 搜索（通用网页索引） | 广筛 30–50 篇题录/摘要 | 结果是搜索引擎摘要，不是全文；单次约返回 8–12 条链接 |
| arXiv abs / ar5iv HTML | 核实题录、读方法与实验章节 | 部分页面截断（超过 100k 字符时只读到前段） |
| PMLR / AAAI / IJCAI / AAMAS 官方页 | venue 与页码核实 | 部分为 PDF，抽取质量不稳定 |
| 出版社页（Springer/ScienceDirect） | 题录核实 | Springer 返回 303 跳登录页；多篇为付费墙，未读全文 |
| GitHub REST API | 版本、许可、最后提交、release | 仅静态元数据，未克隆、未安装、未运行 |
| 机构页（CMU RI） | 报告/学位论文题录 | PDF 为二进制，文本未能抽取 |

未使用：Google Scholar 全文、任何绕过付费墙的渠道、数据集或权重下载、远程主机。

## 检索记录

每条记录格式：时间 → 查询 → 平台 → 返回规模 → 处理。

| # | 查询 | 平台 | 返回 | 处理 |
| --- | --- | --- | --- | --- |
| S01 | memory-driven communication Pesce Montana multi-agent deep reinforcement learning arXiv | 网页 | ~10 链接 | 命中 arXiv:1901.03887，取得 ar5iv 全文，建卡 |
| S02 | multi-agent reinforcement learning experience sharing between agents 2024 2025 | 网页 | ~10 链接 | 命中 3PS、CESDQL、聚类经验共享、Hayashida 2025、YAC 2025；纳入 5 条 |
| S03 | action advising multi-agent reinforcement learning teacher student advice budget | 网页 | ~10 链接 | 命中 Torrey & Taylor 2013、1905.01357、2011.14281、ReBudget、CMU 博士论文 CMU-CS-24-118；纳入 4 条 |
| S04 | learning from failure memory robot manipulation negative experience reuse | 网页 | ~10 链接 | 命中 FEMA、FAR、CAMP、AnchorGUI、"Experience is the Best Teacher"；纳入 3 条，2 条仅记录 |
| S05 | in-context reinforcement learning algorithm distillation decision-pretrained transformer | 网页 | ~10 链接 | 命中 AD、DPT、Vintix II、T2MIR；纳入 4 条 |
| S06 | Shared Experience Actor-Critic for Multi-Agent Reinforcement Learning Christianos 2020 | 网页 | ~10 链接 | 核实 SEAC 题录与代码；读 ar5iv 全文 |
| S07 | learned communication multi-agent reinforcement learning IC3Net TarMAC gating when to communicate | 网页 | ~10 链接 | 核实 IC3Net 1812.09755、TarMAC PMLR v97；纳入 2 条 |
| S08 | neural episodic memory reinforcement learning Neural Episodic Control pritzel 2017 | 网页 | ~10 链接 | 核实 NEC 1703.01988 与 PMLR 页码 |
| S09 | negative transfer multi-agent reinforcement learning transfer learning failure | 网页 | ~10 链接 | 命中 PIMRC 2024 基站控制、DPF-MTMARL、PRIMA 2024、IJCAI 2016 多教师；纳入 IJCAI 2016，其余仅记录 |
| S10 | LLM multi-agent system share failure experience between agents 2025 cross-agent memory transfer | 网页 | ~10 链接 | 命中 Agent KB、MemCo、MACE、EQUIMEM、Mesh Memory Protocol；纳入 Agent KB、MemCo |
| S11 | experience sharing heterogeneous agents reinforcement learning similarity relevance select whom to share | 网页 | ~10 链接 | 命中 Hayashida 2025（ISCIE/SMC）、UMass 动态协同学习、2202.12174；纳入 3 条 |
| S12 | safe reinforcement learning memory of constraint violations avoid repeated unsafe actions | 网页 | ~10 链接 | 命中 MICE（ICML 2025）、Rule-based Shield、PSPO；纳入 MICE、Shield |
| S13 | "experience sharing" multi-agent reinforcement learning survey communication taxonomy 2024 | 网页 | ~10 链接 | 未找到以"experience sharing"为纲的 2024 综述；仅得分类页与二手总结，不作为技术依据 |
| S14 | Melting Pot 2.0 social generalization multi-agent benchmark negative transfer evaluation | 网页 | ~10 链接 | 核实 Melting Pot 2.0 2211.13746；未找到其内置负迁移指标 |
| S15 | Reflexion ExpeL Voyager experience library cross-task reuse language agent | 网页 | ~10 链接 | 命中 ExpeL AAAI-24（2308.10144）、Voyager、Reflexion；纳入 3 条 |
| S16 | multi-agent reinforcement learning episodic memory across episodes teammate adaptation without gradient update | 网页 | ~10 链接 | 命中 EvolveMAS、CooT、LeGEM、Fastap/PACE、TAM 相关；纳入 5 条 |
| S17 | "selective parameter sharing" multi-agent reinforcement learning ICML 2021 SePS | 网页 | ~10 链接 | 核实 SePS 2102.07475 与代码库 |
| S18 | "action advising" OR "advice" multi-agent teaming Shuyang Shi CMU 2025 arXiv teacher student prior knowledge | 网页 | ~10 链接 | 核实 CMU-RI-TR-25-11 为硕士论文；另命中 CMU-CS-24-118 博士论文 |
| S19 | teammate warning communication about failure execution time reinforcement learning learned warning message | 网页 | ~10 链接 | 命中"Agent Malfunction"2025、AAMAS 2026 notifier、MAIC；纳入 2 条 |
| S20 | case-based reasoning reinforcement learning retrieve similar past experience generalize new situation | 网页 | ~10 链接 | 命中 Atzeni ICLR 2022、DECAF、OCaL-RL；纳入 2 条 |
| S21 | COOT coordination transformers in-context multi-agent Overcooked arxiv | 网页 | ~10 链接 | 核实 CooT 2506.23549，ICML 2026 |

另有一次 WebFetch 被用户中断（Explainable Action Advising 全文、negative demonstrations 检索、通信综述检索），因此 EAA 只做了题录级记录。

## 直接打开并核实的一手页面（26 个）

arxiv.org/abs：2311.00865、1901.03887、1611.02779、1903.08254、1905.01357、2507.06229、2610.07376、2603.07110、2601.11953、2202.12174、2506.23549、2211.07882(仅题录)
ar5iv 全文：1901.03887、2311.00865、2006.07169、2205.13718、2011.14281、1905.01357
官方会议页：mlanthology.org/icml/1993、proceedings.mlr.press v97/v199/v216、ijcai.org/Abstract/16/330、ojs.aaai.org 21179
机构页：publications.ri.cmu.edu（CMU-RI-TR-25-11）
GitHub API：facebookresearch/BenchMARL、uoe-agents/epymarl、pytorch/rl、bold-lab-ai/JaxMARL、raw setup.py

## 纳入/排除标准

**纳入（题录+摘要层）**：至少满足其一——(a) 在 agent 之间传递经验/参数/建议/消息；(b) 用记忆或检索复用过去的交互后果；(c) 在学习机制上把"坏结果"转为可复用信号；(d) 执行期不更新参数的上下文适应；(e) 提供本课题必需的训练/评测底座。

**排除**：纯综述与二手博客不作为技术判断依据（只用于发现文献）；仅关键词重合但无跨 agent 或无用例后果的通信工作；无法取得原始页面的条目（已标注 exclude-unverified）。

**未排除但未读全文**：付费墙（CESDQL、3PS、DECAF、Preference-based）、日文（ISCIE）、PDF 解析失败（CMU 硕士论文）。

## 检索偏差与已知缺口

- 只用英文查询。中文文献（除一篇日文期刊外）未系统检索。
- 2024–2026 增量主要靠通用网页索引，可能漏掉未做 SEO 的新预印本；尤其 2026 年 6–10 月的多智能体记忆工作。
- 未做系统性的前后向引用扩展（时间与工具限制）；仅通过搜索引擎摘要间接获得引用关系。这是本轮最大方法学缺口。
- 未检索 ACL/EMNLP 的 LLM-agent memory 专线，仅通过少量命中覆盖。
- 论文数量：CSV 收录 55 条，其中 9 条达到"正文静态阅读"，其余为题录-摘要级。这不是系统综述的覆盖率。

---

# 第二轮（2026-10-29，定向核实）

不再泛增数量，目标是把关键判断落到正文。

## 正文核实（S22–S25）

| # | 动作 | 入口 | 结果 |
| --- | --- | --- | --- |
| S22 | 读 CooT 正文 §3–§6、附录 F | https://arxiv.org/html/2506.23549v3 | 取得：目标为 partner-specific best-response；监督是 ground-truth 最优动作；无失败类别；上下文 5 episodes |
| S23 | 读 FEMA 正文 §3–§4 | https://arxiv.org/html/2603.07110v1 | 取得：失败=过早终止；检索 ℓ₂+ε；不从失败直接更新策略；5 seeds；有不利结果 |
| S24 | 读 MICE 正文 §4.2、§5 | https://arxiv.org/html/2601.11953v1 | 取得：单智能体 CMDP；伪计数核；k=10、ξ=10⁻³；6 seeds；未描述执行期查询 |
| S25 | 读 Agent KB 正文 §3.3、§4.1、§4.4 | https://arxiv.org/html/2507.06229v5 | 取得：成功与失败轨迹都入库存；BM25+MiniLM 混合检索；门控是计划余弦 ≥0.8；有负结果；无 seed/方差 |
| S26 | 读 MemCo 正文 §3.3–§3.5 | https://arxiv.org/html/2610.07376v1 | 取得：Wilson 下界提升；按接收方状态与决策阶段条件化；显式存失败诊断与修复；3 runs |
| S27 | 重读 Pesce & Montana 训练段 | https://ar5iv.labs.arxiv.org/html/1901.03887 | 取得：正文未描述跨 agent 写→读梯度；critic 不接收记忆；记忆每 episode 重置；无 detach 说明 |

## 引用追踪（S28–S30）

用 Semantic Scholar Graph API 取后向引用题录（只取标题/年份/venue，未读正文）。

| 种子 | 返回 | 有用命中 |
| --- | --- | --- |
| SUPER 2311.00865 | 20 条 | INSPIRE (Mach Learn 2026)、Selective Memory Sharing in Multi-Agent LLM Teams (ISDFS 2026)、SCoUT (arXiv 2026)、AERAS (ICRA 2025)、Collaborative Learning in Agentic Systems (arXiv 2025) |
| Pesce & Montana 1901.03887 | 60 条 | MeMAT (Neurocomputing 2026)、CGIBNet (信息瓶颈)、Event-Triggered Communication Network (TNNLS 2021)、Dynamic Size Message Scheduling (TMC 2023)、Discrete-Valued Neural Communication (NeurIPS 2021) |
| CooT 2506.23549 | 0 条 | 索引尚无后向引用 |
| FEMA 2603.07110 | 1 条 | 与本课题无关（Action-Token Pruning） |

**未做**：Agent KB、MemCo、SEAC 的引用扩展；前向引用（citing）与后向引用（references）只做了前者。

## 新增检索（S31–S33）

| # | 查询/动作 | 平台 | 结果 |
| --- | --- | --- | --- |
| S31 | "Boosting Efficient Experience Exchange" sparse-reward MARL 2026 | 网页 | 定位 INSPIRE，*Machine Learning* 2026，DOI 10.1007/s10994-025-06967-y；摘要称 Neighbor Experience Transmitter 用邻居反馈评估经验价值；**正文未取得** |
| S32 | Semantic Scholar DOI 查询 | API | 题录与作者确认；出版社未提供摘要 |
| S33 | （被中断）causal attribution / information bottleneck / event-triggered 三条查询 | — | 改由子 Agent 执行，结果并入本节或 `mechanism-support-matrix.md` |

## 第二轮已知缺口

- INSPIRE 正文未取得——这是当前最影响判断的缺口（若"邻居反馈评估经验价值"属实，问题 B 的空白进一步缩小）。
- CBR（Atzeni et al.）全文未取得，只能按摘要说"只收集正例"。
- 时延（latency）维度：没有一次检索找到报告该指标的论文。
- 中文文献仍未系统检索。

---

# 第二轮补充：三条并行子任务（2026-10-29）

主 Agent 派三个只读子 Agent 分别核实 A（失败识别）、B（跨 agent 适用性）、C（高效传输），各查 60–78 个页面。

## S34 A 线（失败识别与归纳）

- 覆盖：HER、FEMA、MICE、Atzeni-CBR、Shperberg、ShieldPPO、FAR、MAST、AgentDebug、Who&When、MNL、FAIL-Detect、FIPER、RoboMD、CPO、QR-DQN、Mesnard、COCOA、Shapley CF、ETO、UNIQ、FACT、VIPER、PER、Gymnasium time limits。
- **新取得的全文**：Atzeni-CBR（推翻了"未取得"）、Shperberg（§1）、FAR（§4.1、§6）、MAST（§3–4）、MNL（§3.3、App.A.2.1）、FACT（方法节）。
- **新发现且改变判断**：Who&When 的 step 级归因准确率仅 14.2%；FAR 自述 "no principled failure detection mechanism"；Shperberg 的规则由 MAE 给且只能在同构 agent 间共享。
- **未取得**：Grimmeisen 2024（ASME，付费墙）、Xu 2023 ICRA、No Experts No Problem（NeurIPS 2025 全文）、Optimizing RL Using Failure Data（URTC 2024）。

## S35 B 线（跨 agent 适用性）

- 覆盖：INSPIRE、SUPER、Hayashida、MAIC、TarMAC、EAA、CMU-RI-TR-25-11、CACOM、T2MAC、PMAC、XferAgent、RELIC、Agent KB、EF-OnTL、Trust Estimation。
- **新取得的全文**：INSPIRE（Springer HTML，推翻"未取得"）、Hayashida 日文 PDF、MAIC AAAI PDF、CMU 47 页 PDF、CACOM、T2MAC、PMAC、XferAgent。
- **题录更正**：MAIC 是 AAAI-22 不是 2021；EAA 是 ICRA 2023 不是 AAMAS 2022。
- **未取得**：RELIC、Trust Estimation（EUMAS 2026）、EF-OnTL、PIMRC 2024 基站控制（IEEE 418）、CoPS（ICLR 2025 submission，OpenReview 拦截）。

## S36 C 线（高效传输）

- 覆盖：CGIBNet、DVQ、Discrete Messages、IMAC、IB-VQ、Mostaani、SUPER、IC3Net、ETCNet、DSMS、SCoUT、MAIC、SMV、Message Utility Estimation、RAVEN、SeqComm-DFL、Emergence Not Bandwidth、VoI GLOBECOM 2023、Mem0、2601.07978、Agent KB。
- **关键发现**：三条把失真定义为**任务价值损失**的工作（Mostaani、RAVEN、Emergence），推翻第一轮"没有语义对齐先例"的判断。
- **题录更正**：DVQ 是 AAAI-23 不是 AAAI 2022；DSMS 是 TMC 2024 不是 2023；CGIBNet 只有 arXiv 版；SCoUT 真实存在（RLC 2026）。
- **未取得**：ETCNet/DSMS 期刊版、VoI GLOBECOM 正文、Emergence 补充证明。

## 引用追踪汇总

| 种子 | 方式 | 有用命中 |
| --- | --- | --- |
| SUPER 2311.00865 | 后向（Semantic Scholar） | INSPIRE、Selective Memory Sharing (ISDFS 2026)、SCoUT、AERAS、Collaborative Learning in Agentic Systems |
| Pesce & Montana 1901.03887 | 后向 | MeMAT、CGIBNet、ETCNet、DSMS、Discrete-Valued Neural Communication |
| CooT 2506.23549 | 后向 | 0 条 |
| FEMA 2603.07110 | 后向 | 1 条，无关 |

**仍未做**：Agent KB、MemCo、SEAC 的引用扩展；三篇种子的 references（前向）列表。

## 最终计数

CSV 66 条；正文级约 20 篇（含子 Agent 核对）；摘要级 15 篇左右；其余题录。

## 第二轮后仍然存在的缺口

1. CoPS（ICLR 2025 submission）与 "Negative transfer is real" 被 OpenReview 拦截——这两条直接关系到负迁移，值得后续用机构权限补读。
2. MARL 执行期检索时延：仍未找到任何报告。
3. INSPIRE 的 seed 与代码未核实。
4. RAVEN 的完整标题与是否同行评审未核实。
5. 中文文献仍未系统检索。
