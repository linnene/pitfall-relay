# 深读卡片 P14 — Rule-based Shield / Shperberg 2022（唯一明说"失败规则可跨 agent 共享"的工作）

| 项 | 内容 |
| --- | --- |
| 准确标题 | A Rule-based Shield: Accumulating Safety Rules from Catastrophic Action Effects |
| 作者 | Shperberg 等（子 Agent 报告未列全作者，未核实名单） |
| 年份/venue | CoLLAs 2022，PMLR v199:231–242 |
| 链接 | https://proceedings.mlr.press/v199/shperberg22a.html；关联 ShieldPPO: arXiv:2202.09516 |
| 阅读深度 | **正文静态阅读**（§1 与实验；由子 Agent 读取并回报章节与短摘录） |
| 证据定位 | §1（规则来源与限制）；pp.231–242 |

## 机制

- **规则形式**：state-action 谓词（"safety rules (predicates)"）。违反规则的动作用 mask 屏蔽："actions which violate one of the safety rules ... are masked away"。
- **规则来源**：**不是自动归纳**。原文："receive a set of safety rules (predicates) from an oracle (called a Mistake Analysis Entity)"。
- **作者自述限制**："reliance on an external MAE is an obvious limitation"；自动 MAE 列为未来工作。
- **失败定义**：可观测的 catastrophic outcome（碰撞等），归到 (state, action) 等价类。
- **跨 agent**：规则集 "can also be shared between different agents"，但**限定 homogeneous agents**。

## 证据

- 对比 ShieldPPO、CPO、PPO-Lagrangian、PPO-RS；ACL 版本的实验细节本轮未逐表核对。
- seed、算力：未取得。
- 代码：未核查。

## 与本课题的关系

- **相同点**：把失败归纳成结构化规则，并**明确主张规则可以在 agent 之间共享**。
- **关键差别**：
  - 规则由人/预言机给，不是从失败中自动学；
  - 共享只在同构 agent 间声明，**没有异构或"相关但不完全相同"情境的受控实验**；
  - 规则是符号谓词 + 屏蔽，不是可微表征；
  - 没有接收方侧适用性判断。
- **危险等级**：中。它不是"学习式失败表征"，但它占据了"失败规则可跨 agent 共享"这句话。任何主张必须比它多给两样东西：**自动归纳** 和 **异构/近迁移的实证**。
- **可复用**：谓词 + 屏蔽的执行期接口；"规则等价类"作为失败相似度的一种定义。

## 未核实

- 完整作者名单、实验规模、seed。
- ShieldPPO（2202.09516）与本文的关系细节（本轮只读到摘要级描述）。
