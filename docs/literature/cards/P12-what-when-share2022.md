# 深读卡片 P12 — What and When to Share 2022

| 项 | 内容 |
| --- | --- |
| 准确标题 | Collaborative Training of Heterogeneous Reinforcement Learning Agents in Environments with Sparse Rewards: What and When to Share? |
| 作者 | Alain Andres, Esther Villar-Rodriguez, Javier Del Ser |
| 年份/venue | arXiv:2202.12174（2022-02-24）；abs 页标注 under review，**本轮未确认最终 venue** |
| 链接 | https://arxiv.org/abs/2202.12174 |
| 阅读深度 | 摘要级（abs 页已核实） |
| 证据定位 | arXiv 摘要 |

## 机制

- **设定**：异构 agent 在同一环境中各自面对不同实例，只能访问部分状态空间；无先验知识。
- **共享内容**：actor-critic 模型参数 + 内在动机信息。**不是经验表征，不是失败。**
- **使用阶段**：协作训练。
- **执行期**：摘要未讨论执行期共享。
- **失败定义**：无。
- **目标**：内在动机 + 迁移学习。

## 证据

- 环境：改版的 ViZDoom "My Way Home"。
- 基线：独立学习（无共享）。
- 结果：协作框架在少量额外算力下可超过独立学习；作者强调需调好内外奖励权重，否则出现不良行为（未量化）。

## 与本课题的关系

- **相同点**：标题就是本项目的核心问题（what / when to share），且明确以异构性为前提。
- **关键差别**：答的是"参数与内在动机"，不是"失败后果"；无跨局记忆；无接收方适用性判断；无执行期机制。
- **可能致命重叠**：中。它把"what/when"确立为已存在问题——本项目不能把"回答 what/when"本身当作贡献。
- **可复用**：what/when 的问题分解可直接沿用到文献定位轴；其"需要调权重否则行为退化"的观察是负迁移的早期证据。

## 未核实

最终发表venue、全文方法、代码、seed。
