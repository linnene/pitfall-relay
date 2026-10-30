# 深读卡片 P15 — INSPIRE 2026（邻居反馈 + 接收方本地 TD 的适用性基线）

| 项 | 内容 |
| --- | --- |
| 准确标题 | Boosting Efficient Experience Exchange in Sparse-Reward Multi-Agent Reinforcement Learning |
| 作者 | Jianan Zhang, Zaipeng Xie, Nuo Yang, Juguang Jin, Wenzhan Song |
| 年份/venue | *Machine Learning* (Springer) **115:95**, 2026 |
| DOI | 10.1007/s10994-025-06967-y |
| 阅读深度 | **正文静态阅读**（§3.3–3.4、§5.2；由子 Agent 读取并回报章节与短摘录） |
| 证据定位 | §3.3 "aggregates evaluations from visible neighbors"；§3.4 "evaluates each experience using the local value function Q^i to compute its TD error"；§5.2 消融 |

## 机制（两级，与检索摘要给人的印象不同）

1. **Neighbor Experience Transmitter**：对每条经验，聚合**可见邻居的 TD 误差高斯分布统计**（均值/方差），得到一个评分。
2. **接收方二级过滤**：接收方再用**自己的 Q^i** 计算该经验的 TD 误差，决定是否采纳。

- **"neighbor feedback" 的准确含义**：邻居 TD 误差的分布统计，**不是**接收方对经验价值的显式反馈，也不是接收方事后结果。
- **阶段**：训练期。
- **设定**：作者自述 "operates under a homogeneous MARL setting"，完全异构是 future work。
- **失败**：无失败概念，仍是稀疏奖励下的经验共享。

## 证据

- 相对提升（检索摘要口径）：SMAC +13.60%、SMACv2 +12.93%、GRF +37.92%。正文表格本轮未逐项核对。
- 负迁移：仅消融间接，**无专门的逐接收方负迁移指标**。
- seed、算力：本轮未核实。
- 代码/许可：未核实。

## 与本课题的关系

- **相同点**：把"接收方侧的信息"引入经验价值判断——这是问题 B 的核心动作之一。
- **关键差别**：
  - 判断对象是**训练期的转移**，不是失败经验；
  - 接收方信号是自己的 TD 误差，不是"用了这条经验之后失败率是否下降"；
  - 同构 agent；
  - 训练期，非执行期。
- **危险等级**：**高**。它把"接收方信号参与经验筛选"从空白变成已有实现。
- **必须作为基线**：任何声称"接收方条件化适用性"的方法，至少要打赢"邻居 TD 分布统计 + 接收方本地 TD 二次过滤"。
- **可复用**：两级过滤结构；TD 误差的分布统计（而不是单点）作为鲁棒性来源。

## 未核实

- 正文表格的具体数值、seed、代码。
- 高斯统计的具体聚合方式（均值？加权？）。
