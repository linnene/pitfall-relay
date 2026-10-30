# 深读卡片 P09 — Agent KB（第二轮正文核实，已更正）

| 项 | 内容 |
| --- | --- |
| 准确标题 | Agent KB: Leveraging Cross-Domain Experience for Agentic Problem Solving |
| 作者 | Xiangru Tang 等 18 人 |
| 年份/venue | 2025-07-08 提交，v5 2025-10-27；**arXiv 预印本，无会议/期刊 venue** |
| 链接 | https://arxiv.org/abs/2507.06229；正文 https://arxiv.org/html/2507.06229v5 |
| 阅读深度 | **正文静态阅读**（HTML 前 100k 字符，覆盖 §3.3、§4.1、§4.2、§4.4） |
| 证据定位 | §3.3 检索管线与门控；§4.1 构造与评估协议；§4.2 负结果；§4.4 错误分析 |

> **第一轮更正**：此前写"不以失败为标签、摘要未说明失败轨迹是否存储"——正文显示**成功与失败轨迹都被转成结构化经验**，但没有失败标签体系。

## 构造

- 成功与失败运行都被规范化成结构化经验单元（§4.1 "successful and failed runs"）。
- 失败轨迹的后续用途是 pass@2 阶段的 "failure diagnoses"。
- **没有失败分类法或标注规则**；附录 E.2.2 的 "AgentExecutionError" 只是原始日志示例。

## 检索

- 混合检索：**BM25 词法召回**（带 domain/tool 兼容性过滤）+ **all-MiniLM-L6-v2 语义排序**，默认 α = 0.5 融合，top-k 去重。
- 默认 **k = 3**；planning 查询用任务描述，feedback 查询用执行轨迹（§3.3）。

## 是否按接收方条件化

- **不按 agent 身份或模型条件化**。设计目标是 "framework-agnostic abstraction"，靠统一 schema + REST 端点跨框架迁移。
- 唯一的兼容性约束是工具/API 可用性：检索时按 domain/tool 过滤，refinement 时把动作映射到接收框架的可用工具。
- 因此"适用性"在这里是**工具兼容性 + 向量相似度**，不是学习出来的判断。

## disagreement gate

- 只用于 feedback 阶段：比较原计划 ρ 与候选 ρ' 的 **all-MiniLM-L6-v2 嵌入余弦相似度**，要求 ≥ β = 0.8。
- 输入是两个计划本身，**不是** agent 置信度、投票或不确定性。
- 低于阈值则拒绝该 refinement。

## 证据

- 基准：GAIA 165、HLE Bio/Chem 149、GPQA 198、SWE-bench Lite 300（§4.1）。
- 协议：每个实例跑**三次顺序 pass**——pass@1 无标签，pass@2 加入首轮失败诊断，pass@3 用扩展检索池重访未解决实例。**注意：这里的 pass@k 是顺序知识库增强，不是独立重复采样。**
- 默认 GPT-4.1，temperature 1.0，top-k 3。
- 数字（摘要）：smolagents pass@3 55.2%→73.9%；OpenHands SWE-bench pass@1 24.3%→28.3%。
- **危害测量**：错误分析统计新增错误 15（GPT-4.1）/11（Claude-3.7），同时给出修正的基线错误数，净减少。
- **负结果**：OWL 在 GAIA Level 3 的 pass@1 从 30.8 降到 23.1（表 2b）。
- **没有专门的"有害经验隔离"实验**，破坏性不是主要终点。
- **无 seed、无置信区间、无 run-to-run 方差报告。**

## 与本课题的关系

- **相同点**：一个 agent 的经验帮助另一个 agent 少犯错，且明确量化。
- **关键差别**：
  - 文本知识库，非神经经验表征；
  - 检索在 planning/feedback 层，不是执行期逐步决策；
  - 无发送方/接收方权限与预算；
  - 适用性 = 相似度 + 工具兼容性，非学习判断；
  - 无"相关但不相同情境"的受控评测。
- **它证明的事**：跨 agent 经验共享**在 LLM 路线已被做到并量化**；并且**已经出现了负结果**（OWL 下降），说明负迁移是真实风险。
- **可复用**：三 pass 的评估协议（可作为"经验是否真的被用上"的检验）；错误分析统计"新增错误"的口径；disagreement gate 作为启发式对照。

## 未核实

- 代码与许可证；失败诊断的具体生成方式。
