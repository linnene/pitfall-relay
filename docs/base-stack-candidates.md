# 候选 base 技术栈

性质：**静态核查（版本、许可、机制密度）+ 2026-10-30 在本机 macOS arm64 的实测锁定**。框架选择、接入难点与算力判断仍属静态分析；本机实测只覆盖 §L1 的版本解析与 import/配置检查。**Linux、wyb、CUDA、任何训练或 rollout 均未验证**。版本与许可的原始来源为 GitHub REST API、仓库内 `setup.py` 原文，以及 `docs/development.md` 记录的实测解析结果。

原则：训练框架 ≠ 算法 ≠ 创新。框架只负责把实验跑起来；本项目的可辩护部分必须留在自己的代码里。

---

## 分层候选

### L1 训练框架

| 候选 | 版本/状态（2026-10-29 核查） | 许可 | 判断 |
| --- | --- | --- | --- |
| **BenchMARL** | release 1.5.2（2026-02-07），最后 push 2026-02-07，678 stars，未归档 | MIT | 首选骨架；但见下方版本陷阱 |
| **EPyMARL** | 最后 push 2024-09-24，741 stars | Apache-2.0 | 离散协作任务的备选；维护活跃度已下降 |
| **TorchRL** | 0.14.1（2026-10-10），当日仍在提交 | MIT | BenchMARL 的底层；版本必须锁 |
| **JaxMARL** | 最后 push 2026-10-26，org 为 bold-lab-ai，868 stars | Apache-2.0 | 仅在吞吐成为瓶颈且愿意迁移 JAX 时 |
| MAPPO 官方 on-policy | 本轮未核查版本 | 未核查 | 只作为算法参考，不当作工程底座 |

**版本锁定：已从静态推断升级为实测（2026-10-30）**。macOS arm64 / CPython 3.11.18 / conda 环境下实际解析成功的组合：

| 包 | 锁定版本 | 约束来源 |
| --- | --- | --- |
| benchmarl | 1.5.2 | 当前 release；PyPI 仅 sdist，构建为 `py3-none-any` 纯 Python wheel |
| torchrl | 0.11.1 | BenchMARL 要求 `>=0.10,<0.12` |
| tensordict | 0.11.0 | torchrl 0.11.1 要求 `>=0.11.0,<0.12.0`，该区间内只有 0.11.0 |
| torch | 2.10.0 | 与 torchrl 0.11.1（2026-02-05）同期；更新的 torch 未被 BenchMARL 约束覆盖 |
| torchvision | 0.25.0 | 声明 `torch==2.10.0`，其他组合解析失败或 import 失败 |
| av | 13.1.0 | BenchMARL 无条件列 `av<14`；13.1.0 有 cp311 macOS arm64 wheel，**未触发源码编译** |
| hydra-core | 1.3.2 | BenchMARL 配置层 |
| numpy | 2.4.6 | 实测可解析 |

完整命令与已验证/未验证状态见 [`development.md`](development.md)。

**仍未验证**：Linux、wyb、CUDA、任何训练或 rollout。`av` 与 `benchmarl` 在 Linux 上的解析结果**不能**由 macOS 结果外推；到 wyb 上仍需按目标 Python 版本实测一次。

### L2 策略算法

| 候选 | 适用 | 备注 |
| --- | --- | --- |
| 循环 MAPPO | 连续/离散均可，CTDE | 起点；有官方实现可对照 |
| QMIX / VDN / IQL | 离散动作 | EPyMARL 原生支持；LeGEM 用 10 seeds 做过这类基线 |
| SEAC / SUPER 风格 | 训练期共享对照 | 作为**基线**而不是方法 |

### L3 经验编码 / 记忆

| 候选 | 文献依据 | 状态 |
| --- | --- | --- |
| 读/写门控共享记忆 | Pesce & Montana 2020 | 需自己实现；无公开代码 |
| afterstate 图记忆 | LeGEM 2022 | 结构可借鉴，代码未定位 |
| 非参数近邻记忆（DND 式） | NEC 2017 | 实现成熟，但需处理连续状态 |
| 上下文 Transformer | CooT 2025/26 | 上下文长度与冷启动协议可直接借 |
| 变分上下文推断 | PEARL / VariBAD | 用于"适用性"潜变量，非失败专用 |

### L4 接收与相关性机制

| 候选 | 依据 | 说明 |
| --- | --- | --- |
| 接收方注意力 | TarMAC 2019 | 隐式相关性，最省事 |
| 发送方打分 | SUPER 2023 | 必须作为基线，不是方法 |
| 相似度阈值 | Hayashida 2025 / 聚类 2024 | 必须作为"无学习"基线 |
| 分歧门控 | Agent KB 2025 | 负迁移防护的启发式对照 |
| bandit 探测队友能力 | CMU-RI-TR-25-11 | 接收方建模的强基线 |

### L5 任务 / 环境

| 候选 | 提供 | 与本课题的匹配 |
| --- | --- | --- |
| Melting Pot 2.0 | substrate + background population，测社会泛化 | 适合"对未见队友泛化"，但算力需求大（原文每场景 1e9 steps） |
| Overcooked（CooT/JaxMARL 均有） | 协作、部分可观测、易做出隐藏风险 | 通用性强，但已有大量基线，需避免与 CooT 直接同台 |
| LBF / RWARE | 离散协作、稀疏奖励 | SEAC 的标准任务，做共享对照方便 |
| PettingZoo Pursuit / Battle | SUPER 的任务 | 便于直接复现其 share-all 负结果 |

任务尚未选定，本节不给结论。选任务的硬条件仍是 research-brief 的 decision gate：普通位置/状态警告能解决的任务不能用。

---

## 三套可行组合

### 组合 A（推荐起步）：BenchMARL 1.5.2 + torchrl 0.11 链 + 循环 MAPPO + 自定义消息模型

- **可原样复用**：训练循环、采样、日志、评测、TensorDict 数据管线、序列模型（GRU）路径。
- **需自己写**：经验编码器、消息接口（agent×team×time 维）、接收方条件化、触发器；以及跨 group 的梯度路径。
- **方法假设**：CTDE 下 actor 只用执行期可获得的信息；消息在一个 group 内传递。
- **已知难点**（承接 Finn 的底座评估，本轮核查方向一致）：
  - GNN 接收同组 agent 特征，**不等于通信协议**；全连接图会让 actor 免费获得全部信息。
  - 缓存/detach 的经验编码器**不会**从接收方损失获得梯度，表面上能跑但学不到东西。
  - GRU 的训练序列切段与采集时携带 hidden 是两条路径，episode 重置若处理不当会串历史。
  - 训练会过滤部分 `info` 键；经验输入必须显式进入保留的数据与输入规格。
- **版本兼容风险**：中-高。torchrl 被固定在旧区间，torch/tensordict 需一起锁。
- **静态成本**：一个 conda/venv 环境 + 一次最小闭环（采样→更新→评测→存读 checkpoint）；未实测。
- **算力**：未实测，不给承诺。

### 组合 B（离散任务，改动最小）：EPyMARL + QMIX/VDN/IQL + EpisodeBatch 级消息注入

- **可原样复用**：EpisodeBatch 天然带 agent 维，跨 agent 取数据最直观；GRUCell、BasicMAC 结构清晰；有大量离散基线。
- **需自己写**：控制器输入拼接、消息模块注册、保存/加载。
- **方法假设**：离散动作、同构 agent。
- **难点**：默认控制器按 agent 展平输入；通信与参数注册要扩展；依赖较旧（Python/torch 版本需试）。
- **维护风险**：仓库最后 push 2024-09-24，长期无人维护。
- **算力**：未实测。

### 组合 C（吞吐受限时）：JaxMARL + Overcooked/LBF

- **可原样复用**：JAX 环境与训练可一起编译；Overcooked 实现现成。
- **需自己写**：可变长经验历史必须固定形状 + mask；消息模块要写成 jit 友好的纯函数。
- **方法假设**：愿意迁移到 JAX 技术栈；现有 Python 环境不能自动获得编译收益。
- **风险**：迁移成本高；调试梯度路径更难（jit 内的 detach 更隐蔽）。
- **算力**：未实测。

**不建议**：以 IC3Net 原仓库作为基础设施（依赖 torch 0.4/gym 0.9）；以 CoDreamer/DMAWM 作为底座（与本问题无必然关系，且会引入世界模型这一未决架构假设）。

---

## 第二轮：机制结论对底座选择的影响

以下三点来自 `docs/literature/revision-notes.md` 与 `mechanism-support-matrix.md`，只有它们影响选型，其余框架信息不重复展开。

1. **跨 agent 梯度路径是选型的第一约束**。Pesce & Montana 的正文未描述"发送方写入经接收方损失回传"（`cards/P04`）。若本课题要主张这条路径，框架必须允许**不同 group 的模型出现在同一计算图里**。
   - BenchMARL 按 group 组织模型与损失，做这件事需要自定义损失并显式验证梯度范数，**最容易出现"表面能跑、实际 detach"**。
   - EPyMARL 的 EpisodeBatch 把全体 agent 放在一个 batch 里，跨 agent 计算图更直接，但仓库最后 push 为 2024-09-24。
   - **这是唯一一条可能改变首选框架的机制证据**：如果跨 agent 梯度是核心主张且验证成本过高，应转向 EPyMARL 或放弃该主张（退回端到端但按接收方 detach 的版本，并在论文中声明）。
2. **时延从未被文献报告**（`mechanism-support-matrix.md` C4）。若执行期使用是主张之一，必须在真实候选任务上自测"一次前向 + 一次检索"的时延，不能引用他人数字。
3. **失败定义决定环境接口**：若选"约束违反"，需要代价通道（MICE 式 CMDP）；若选"过早终止"，只需要终止标志（FEMA 式）。这会影响环境选择，但不影响框架选择。

其余框架比较结论（BenchMARL 版本锁定、GNN ≠ 通信协议、GRU ≠ 长记忆）保持不变。

## 采用前的最小验收（建议，未执行）

1. 独立环境锁版本，跑通原始循环 MAPPO 的采样→更新→评测→checkpoint 存读闭环。
2. 接入最小消息接口，核对 agent/队伍/时间维，确认 episode 重置不串历史、未来结果不泄漏。
3. 单独验证经验编码器到**接收方**损失的梯度确实存在（打印梯度范数，而不是看 loss 下降）。
4. 屏蔽实验：把消息置零/打乱后性能应显著下降，否则信息通道是假的。
5. 在真实候选任务上记录峰值显存、环境步与 agent 步口径、吞吐，再谈预算。

**不做承诺**：本轮没有证据表明任何一套组合比另一套跑得更快或更省卡。

---

## 来源

- BenchMARL 仓库与 setup.py：https://github.com/facebookresearch/BenchMARL
- EPyMARL：https://github.com/uoe-agents/epymarl
- TorchRL：https://github.com/pytorch/rl
- JaxMARL：https://github.com/bold-lab-ai/JaxMARL
- Melting Pot 2.0：https://arxiv.org/abs/2211.13746
- 机制来源见 `docs/literature/cards/` 与 `docs/literature/nearest-neighbor-matrix.md`。
