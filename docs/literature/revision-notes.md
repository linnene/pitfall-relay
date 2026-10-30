# 修订说明（第二轮）

日期：2026-10-29（第二轮）。修订对象：第一轮 `README.md`、`nearest-neighbor-matrix.md`、`research-opportunities.md`、`cards/`、`search-log.md`。

第一轮版本不再可用作结论依据；本文件记录**哪些说法被撤回、改成什么、依据是什么**。旧文件已就地更新，历史措辞记录在此。

---

## 总体问题

第一轮把 52 条题录级条目与 4 篇正文级阅读混在一张表里，并由此推出若干**存在性结论**。这是证据强度越界。本轮的规则：

| 允许的说法 | 不允许的说法 |
| --- | --- |
| 论文 P 在第 S 节明确做了 X | 没有工作做 X |
| 论文 P 在已读范围内未描述 X | 论文 P 没有 X 机制 |
| 本次检索（截至 2026-10-29）未发现 X | X 仍是空白 |
| 该机制需要额外假设 H | 该机制不可行 |

以下所有更正都按这个规则重写。

---

## 逐条更正

### C1. "52 条中仅 4 篇全文、16 篇摘要核实" 的表述与后续引用脱节

- **旧**：`README.md` 承认了这一比例，但同一份文件给出"唯一余量""已被占/仍空着"的结论表，`research-opportunities.md` 进一步给出"切口 3 不成立时后两步都不用做"的推理链。
- **新**：证据强度声明前置到每份文件顶部；结论一律改写为检索性陈述。
- **影响文件**：README、nearest-neighbor-matrix、research-opportunities、search-log。

### C2. "12 篇核心深读卡片" 的措辞

- **旧**：README 标题写"12 篇核心深读卡片"，容易被读成 12 篇全文精读。
- **新**：明确分级——**正文静态阅读 4 篇**（SEAC、SUPER、Pesce & Montana、LeGEM），**摘要核实 8 篇**，其余仅题录。第二轮新增 CooT、FEMA、MICE、Agent KB、MemCo 的正文核实，正文级数量升到 9，但仍不等于"12 篇全文精读"。
- **影响文件**：README、全部卡片。

### C3. "CooT 只差一个失败维度"

- **旧**：`nearest-neighbor-matrix.md` 与 `cards/P06` 称 CooT 与本研究"只差失败维度"，并把它的危险等级定为"极高"。
- **新（已按正文核实）**：**这个说法错误**，危险等级下调，但改为"必须正面处理的结构性近邻"。逐项对比如下（来源：arXiv 2506.23549v3 HTML）：

| 维度 | CooT 的实际做法 | 与本研究的关系 |
| --- | --- | --- |
| 问题目标 | 不等同于任务求解，而是 **partner-specific best-response**，形式化为 Hidden-Utility Markov Game；§3.1 原文 "shift the objective from general task-solving to partner-specific best-response" | 目标函数不同：本研究是"减少重复失败"，CooT 是"对隐藏效用伙伴做最优响应" |
| 信息权限 | 部署时只有"context C, current state s_t, and a short history of prior states"；上下文记录的是自己的 "transition (s_t,a_t,r_t)"；**没有显式的伙伴观测/动作通道**（§4.3 与 §4.4 措辞有歧义，§6 称 "action histories without explicit communication"） | 双方都是"只能看到自己那条通道"，这点一致 |
| 数据来源与监督 | 36 条 best-response 策略，每条配一个 behavior-preferring 伙伴，用 MAPPO 训练；监督标签是 "ground-truth best-response action"；只用 best-response 轨迹训练 | **这是最大差别**：CooT 需要专家最优动作标签，本研究假设只有自己产生的不良后果 |
| 时序 | 上下文窗口 5 个 episode，冷启动用 T 条空轨迹，最旧被挤出；评估跑 50 个连续 episode | 一致（跨局、无梯度） |
| 学习目标 | 监督序列建模，交叉熵 L = −log p̂_h(â)，**没有奖励最大化项** | 不同：CooT 不优化回报，本研究必须从后果中学习 |
| 失败表征 | **完全没有**。失败不是监督类别；损失只用最优动作标签；伙伴的隐藏奖励含惩罚（如 "large penalty for picking up dishes"），但 "observable only by π_w"，CooT 收不到 | 这是真实差异，但不是"只差一维" |
| 基线 | BC, BC-RNN, MEP, HSP, HSP-ft, HSP-meta, PACE；附录另比 GAMMA、PLASTIC | 可复用为对照模板 |

- **结论**：CooT 与本研究**共享"执行期无梯度 + 跨局上下文 + 多智能体"三项外部形态**，但**学习信号来源完全不同**（专家最优动作监督 vs 自身不良后果）。说"只差失败维度"会低估差异，也会在答辩时被一句"你的监督从哪来"问倒。
- **影响文件**：`cards/P06-coot2025.md`、`nearest-neighbor-matrix.md`。

### C4. Pesce & Montana 的跨 agent 梯度

- **旧**：`cards/P04` 写"接收方损失未必能训练发送方编码器"，`base-stack-candidates.md` 写"缓存不等于学会总结经验"。
- **新（正文核实，ar5iv §训练段落）**：论文称所有策略参数端到端由 actor-critic 更新训练；**读到的正文没有描述 agent j 的写入会通过 agent i 的损失回传梯度**；集中式 critic 接收全体观测与动作，**不接收记忆**；记忆每个 episode 重置，agent 按顺序读写。正文也**没有**显式 detach 或 stop-gradient 的说明。
- **准确表述**：该论文的公开正文**未描述**跨 agent 的写→读梯度路径；不能据此断言该路径不存在或不可实现。本研究的"发送方编码器经接收方损失获得梯度"因此是**待核查的差异**，不是已证空白。
- **影响文件**：`cards/P04`、`base-stack-candidates.md`。

### C5. FEMA 的机制（原卡片为推测）

- **旧**：`cards/P07` 写"推测是检索式、不改策略参数"。
- **新（正文核实，arXiv 2603.07110v1）**：
  - 失败 = **过早终止**，没有学习式检测器；§3.2 原文 "an episode is defined as a failure episode when the training rollout terminates accidentally"。
  - 记忆存失败轨迹**最后 K 步**，条目为 (z_s, a, φ(s,a), H)，H 是该转移的蒙特卡洛回报；按 M 个新失败事件周期性刷新。
  - 检索：状态嵌入的 **ℓ₂ 距离 + 阈值 ε** 初筛，再取"H_i 最低的 top-O 状态-动作对"，用联合嵌入距离 D_i 与风险头 ρ_i 打分 S_i = D_i − λρ_i。
  - **不从失败直接更新策略**：§3.3 用检索结果"directly intervene in the robot's interaction"，重新给候选动作排序；风险头与联合嵌入用 MSE 在失败记忆上训练，策略仍由原 RL 算法更新。
  - 单智能体；5 seeds；基线 SAC/EMAC/SAC+FEMA/PPO/PPO+FEMA/CrossQ/CrossQ+FEMA。
  - 报告了不利结果：EMAC 全面不佳；CrossQ+FEMA 在 Ant 仅 8.33%；PPO+FEMA 在 Humanoid 仅 1.98%（方差大）；ε=0.50 会偏向保守行为。
- **影响**：`cards/P07` 重写；`research-opportunities.md` 里"FEMA 已做完整因果链"改为"FEMA 已做**单智能体**的失败记忆—检索—动作干预链，且不通过梯度更新策略"。

### C6. MICE 的实际范围

- **旧**：卡片对多智能体、执行期使用的表述含糊。
- **新（正文核实，arXiv 2601.11953v1）**：单智能体 CMDP；记忆是"flashbulb memory" M，只存**外生代价 > 0 的不安全状态**；内在代价 c^I_t = sqrt(Σ_{s^m∈N_k} K(f(s_t), f(s^m)))，K(x,y)=ξ/(l²+ξ)，自 NGU 取 ξ=10⁻³，k=10；记忆喂给"外生-内在代价价值函数"，正文**未描述执行期查询**；6 seeds，10⁷ steps；基线 PID Lagrangian/CUP/Saute/Simmer；定理依赖有限 MDP、查表、无限更新假设。
- **影响**：`cards/P08` 更新；矩阵中"跨 agent"一栏保持"未说明"。

### C7. Agent KB 的失败处理与门控

- **旧**：卡片写"不以失败为标签；摘要未说明失败轨迹是否单独存储"。
- **新（正文核实，arXiv 2507.06229v5）**：
  - **成功与失败轨迹都被转成结构化经验**（§4.1 "successful and failed runs"），失败的后续用途是 pass@2 的 "failure diagnoses"；但**没有失败标签体系或标注规则**（§E.2.2 的 "AgentExecutionError" 只是示例）。
  - 检索：BM25 词法召回（带 domain/tool 兼容性过滤）+ all-MiniLM-L6-v2 语义排序，α=0.5 融合，默认 k=3。
  - **不按 agent 身份或模型条件化**；只做"工具/API 兼容性"过滤。
  - disagreement gate：比较原计划 ρ 与候选 ρ' 的 **embedding 余弦相似度 ≥ 0.8**，低于阈值就拒绝；输入是两个计划，**不是** agent 置信度或投票。
  - 有初步危害测量：新增错误 15（GPT-4.1）/11（Claude-3.7），并报净减少；有负结果 OWL 在 GAIA L3 pass@1 从 30.8 降到 23.1；**无专门隔离有害经验的实验**。
  - **无 seed、无置信区间、无 run-to-run 方差**；其 pass@k 是顺序知识库增强，不是独立重复采样。
- **影响**：`cards/P09` 更新。

### C8. MemCo 的机制（第二轮新增正文核实）

- 局部记忆 $M_i^l=[G,G,P,E,R]$：状态-动作-反馈图、模式记录（含 "workflows, action preconditions, repairs, and failures"）、状态-动作-结果关系、具体案例与反思（含 "failure diagnoses, and recoveries"）。
- 全局记忆只存提升后的抽象记录，字段为 "record type, applicability conditions, action sequence or relation, and outcome"。
- 提升准则：按类型与内容归并 → 结构/低信息过滤 → 用**按 episode 去重**的证据计数算单侧 Wilson 下界，要求 W ≥ λ = 0.35；矛盾优先；证据变化会重新评估并可能删除。
- 检索：查询由目标、当前观测、可用动作、历史构成；局部/全局各 k=5；打分 r = r_x + r_p + r_g（状态动作匹配 + 任务进度/决策阶段匹配 + 目标相关角色匹配）；**适用性是硬过滤**（条件矛盾或动作不可用直接排除）。
- **明确按接收方当前状态与决策阶段条件化**，并对过期/软冲突的局部信息降权。
- 失败：模式层、案例层、图边、负证据 n⁻ 四处都存；stall 标记 n⁰ 不计入 Wilson 样本量。
- 3 runs 的 mean±std；基线 ICL/A-Mem/MemRL/MemSkill/G-Memory；局限：冷启动依赖初始轨迹、只测文本/符号环境。
- **对第一轮结论的影响**：这是**跨 agent 经验共享 + 接收方条件化检索 + 显式失败记录**的组合，且是 2026 年 10 月的工作。它把"接收方条件化"从"空着"降级为"在 LLM 文本路线已有实现"。

### C9. "选择性共享优于全共享" 是无条件结论

- **旧**：矩阵与 README 直接写"选择性共享 > 全共享"。
- **新**：SUPER 自身数据显示结论**依赖任务**——Pursuit 148.27 vs 181.43、Battle 9.25 vs 9.46（全共享更差），但 **Adversarial Pursuit 全共享更好**（−568.91 vs −719.78）。准确表述：在全共享有损的任务上，选择性共享可以追回损失；在共享有益的任务上，过度筛选会损失收益。这正好说明"适用性判断"不是可选项。

### C10. "训练期 TD 误差 = 失败定义"

- **旧**：`cards/P03` 表述为"用 TD 误差代替失败定义"。
- **新**：SUPER 的 TD 误差是**共享有用性**的代理，不是失败标签；论文没有失败概念。两者不能混为一谈。准确表述：SUPER 用 |TD| 决定"哪些转移值得传"，与"这条转移是否代表一次失败"无关。

### C11. "等比特预算 = 等语义信息量"

- **旧**：`research-opportunities.md` 写"等带宽对照"，未区分量纲。
- **新**：拆成四个量纲——消息字节/比特、通信频次、编码与推理时延、样本与墙钟成本。等比特不等于等语义信息量；对照必须同时报告"传了多少比特"和"传了什么信息"。详见 `mechanism-support-matrix.md` 的 C 部分。

### C12. "切口 3 不成立后都不用做" 的顺序依赖

- **旧**：`research-opportunities.md` 给出 3 → 1 → 2 的硬顺序，并称第一步不成立时后两步不用做。
- **新**：撤回该硬结论。三个问题是并列的技术问题，不是串行依赖。评测协议（C 的一部分）会影响 A/B 的可解释性，但 A 的失败归纳与 B 的适用性判断可以独立成立并独立验证。

### C13. CBR 不能用来证明"完整失败机制已做"

- **旧**：矩阵把 Atzeni et al. 2022 列为"检索相似经历以泛化到新情境的最直接证据"。
- **新（第二轮已取得正文，§3 Retain、§4.1–4.2）**：
  - retain 的触发是 **r_t > 0**，原文 "whenever r_t > 0, a retain function is used to select which of the past executed actions"；case 是 context-action 对；**只存正例，失败不进入 case base**。
  - 相似度 = 共享 VQ code 的比例；**复用时不更新参数**，训练时更新 retriever 与 policy。
  - 因此它只能支持"检索式复用可以改善 OOD 泛化"，**不能**支持"失败经验检索已做"，也不能反推"失败经验检索不重要"。
- **影响**：`cards/` 中涉及 CBR 的表述；`nearest-neighbor-matrix.md` 族 E。

### C15. 新增一条关键的跨 agent 线索：Shperberg 的规则共享

- Rule-based Shield（PMLR v199:231–242，CoLLAs 2022）的规则是 **state-action 谓词**，由**人类/预言机 MAE**（Mistake Analysis Entity）提供；论文自述 "reliance on an external MAE is an obvious limitation"，自动 MAE 留给未来工作。
- 但论文明确说规则集 "can also be shared between different agents"，**限定 homogeneous agents**。
- **意义**：这是本轮找到的**唯一**把"失败归纳成结构并跨 agent 共享"直接写出来的工作。它有两个硬约束：规则靠人给、共享只在同构 agent 间。
- **不能由此推出**：异构 agent 或"相关但不完全相同"情境下的失败规则迁移已被验证——本轮未发现这样的受控实验。
- **影响**：`nearest-neighbor-matrix.md` 族 E 与 `mechanism-support-matrix.md` A/B 两节。

### C14. 新增关键近邻：INSPIRE（2026）

- "Boosting Efficient Experience Exchange in Sparse-Reward Multi-Agent Reinforcement Learning"（Zhang, Xie, Yang, Jin, Song；*Machine Learning*, 2026；DOI 10.1007/s10994-025-06967-y）。
- 据检索摘要：框架名 INSPIRE，把团队经验拆成个体经验，把选择性共享限制在局部邻居内；**Neighbor Experience Transmitter 用邻居反馈评估经验价值**，做"定向交换高价值经验"。SMAC +13.60%、SMACv2 +12.93%、GRF +37.92%（相对提升）。
- **证据层级**：仅检索摘要，**未取得正文**。已派生子任务核实，见 `search-log.md`。
- **影响**：如果"用接收方/邻居反馈评估经验价值"确已在训练期实现，那么问题 B 的空白进一步缩小到"执行期、按接收方、失败特异"三者叠加。第一轮把它当作空白是过早的。

---

## 结论层面的重写

| 第一轮说法 | 第二轮改写 |
| --- | --- |
| "唯一余量是发送方失败→接收方适用性→执行期使用" | "在本次检索与已读范围内，未发现同时满足『发送方自身失败驱动』『接收方条件化适用性』『执行期无梯度使用』三条的工作；这是检索结论，不是空白证明" |
| "已被占/仍空着" | "已核实有实现 / 已读范围未描述 / 检索未发现 / 未取得全文"四档 |
| "切口 2 依赖切口 1" | 三个技术问题并列；工程上共享底座，科学上不互为前提 |
| "FEMA 已做完整因果链" | FEMA 做了单智能体的失败记忆—检索—动作干预链，且不更新策略参数 |
| "CooT 只差失败维度" | CooT 在学习信号来源（专家最优动作）上与本研究根本不同；共享的只是部署形态 |

---

## 子 Agent 返回后的追加更正（C16–C22）

### C16. INSPIRE 正文已取得，机制与第一轮推测不同

- 题录更正：Zhang, Xie, Yang, Jin, Song, *Machine Learning* **115:95**, 2026。
- 实际机制是**两级**：Neighbor Experience Transmitter 用**邻居 TD 误差的高斯分布统计**聚合评分；接收方再用**自己的 Q^i** 算 TD 误差做二次过滤（§3.4 原文 "evaluates each experience using the local value function Q^i to compute its TD error"）。
- 所谓 "neighbor feedback" **不是**接收方对经验价值的显式反馈，是邻居 TD 误差的分布统计。
- 论文自述 "operates under a homogeneous MARL setting"，**完全异构是 future work**。
- **影响**：第一轮"接收方条件化是空白"已被彻底推翻；新的准确表述见 C17。

### C17. 问题 B 的重新定位：五级适用性信号谱

现有适用性信号按"用到的接收方信息"分五级：
1. 发送方 TD 误差（SUPER；INSPIRE 一级）——无接收方信息。
2. 成对相似度（Hayashida：观测 L2 + 策略全变差 + 未学习度熵；CMU：策略相似度 + CE 阈值）。
3. 接收方自己的 TD 误差（INSPIRE 二级）。
4. 接收方不确定性下降（T2MAC, AAAI-24）。
5. 接收方历史结果（XferAgent, arXiv 2610.19581，**执行期**，且用 leave-one-agent-out 排除自身结果）。

**新的空白表述**：不是"没人做逐接收方判断"，而是**未发现**同时满足「以发送方失败为对象 / 用接收方自身后续结果校准 / 执行期无梯度 / 异构并报告逐接收方负迁移」的工作。
**另一点**：现有 sender TD 误差全部被当作"有用性"的**正**代理，**未发现**把"发送方失败严重程度"当**负信号**的机制。

### C18. 问题 C 的重新定位：任务价值失真已有先例

第一轮说"没有工作把传输成本与决策收益放在同一坐标系"，**错误**。已有三条：

| 工作 | 做法 |
| --- | --- |
| Mostaani et al. (2005.14220) | distortion = 接收方 value loss \|V*(s)−V*(h(s))\|，约束 log2\|C\| ≤ R（2 bit/channel use） |
| RAVEN (2609.37566) | 接收方 context 内的 action-value 失真，4 符号 = 2 bit/message；报告 FLOPs −82.2~96.8% |
| Emergence, Not Bandwidth (2609.34373) | 理论：最优消息是**接收方 advantage 的统计量**，不是发送方观测；固定 2.000 bit/decision |

**准确结论**：这些先例都在**通信学习**这一大类里，**不在"失败经验传输"**。所以缺口是对象而非方法：把 value-distortion 的度量用到"失败经验"这个内容上，本次检索未发现先例。

### C19. 第一轮"时延从未被报告"是错的

- Mem0：search p50 0.148 s / p95 0.200 s；full-context total p95 17.117 s。
- 2601.07978：200 ms 延迟档、8 Mbit/s、50 ms jitter；vector LTM 仅 4%–5% 延迟惩罚。
- RAVEN：FLOPs 比 NDQ 少 82.2–96.8%。

**准确结论**：时延在 LLM-agent 记忆/检索与通信架构里有报告；在 **MARL 执行期经验检索**上没有。数字不能跨域迁移，协议可以。

### C20. 题录更正（第一轮写错）

| 论文 | 第一轮 | 实际 |
| --- | --- | --- |
| MAIC | AAAI 2021 | **AAAI-22**, pp.9466–9474 |
| Explainable Action Advising (EAA) | AAMAS 2022 | **ICRA 2023**（arXiv 2211.07882 的 comment 已写明） |
| Adaptive Discrete Communication Bottlenecks (DVQ) | AAAI 2022 | **AAAI-23**, 37(7):8825–8833 |
| Dynamic Size Message Scheduling (DSMS) | TMC 2023 | **TMC 2024**, 23(12):15080–15097 |
| CGIBNet | 有期刊版 | **仅 arXiv 2112.10374**，未发现期刊版 |
| SCoUT | 待核实 | **真实存在**：RLC 2026 + arXiv 2603.04833 |

### C21. Hayashida 全文已读（此前仅摘要）

- 日文 PDF §3.1–3.4、§4.2；10 trials。
- 机制是**有序对 i←j 的共享比例 X_{i,j}**：观测 L2 距离 + 匹配观测上的策略全变差距离 + "未学习度"熵。
- 有负迁移证据：固定比例共享在环境 2 造成暂时退化；抑制不相似共享可避免恶化。

### C22. CMU 硕士论文已读（此前 PDF 未解析）

- 47 页；§3.2–3.3、§4，pp.10–14。
- 机制：每个学生一个子任务分配；用学生策略与教师子任务策略的负交叉熵作 policy similarity，超过阈值 η 就不发建议（Eq.3.8）。
- **全文未出现 "negative transfer" 或 "applicability" 术语或指标**——这是"未描述"，不是"没有"。
- 它是**逐学生、逐状态的建议门控**，但判断的是"学生是否已会某子任务"，不是"发送方失败对接收方是否有价值"。

### C23. 已读正文但未见适用性判断的清单（区分"未描述"与"没有"）

SUPER（接收方完全不参与）、MAIC（逐接收方生成但依据是队友模型/协调收益）、TarMAC（注意力是当前消息相关性）、PMAC（个性化是架构设计）、CACOM（端到端发送决策，无显式适用性分数）、Agent KB（门控比较两个计划，不涉及接收方身份）。

---

## 仍然未解决、需要在正文层面确认的问题

1. INSPIRE 的 Neighbor Experience Transmitter 是否真的用**接收方**信号、是否逐接收方定制、训练期还是执行期。
2. Pesce & Montana 是否在补充材料或代码中实现了跨 agent 的写→读梯度（正文未描述）。
3. 是否存在**执行期**、**按接收方**、**以失败为唯一内容**的传输工作（本轮未发现）。
4. CBR 全文是否真的只存正例（摘要如此，正文未取得）。
5. Agent KB / MemCo 的"适用性"是否只是检索相似度 + 硬过滤，而不是学出来的判断（正文显示是启发式，未做学习式判断）。
