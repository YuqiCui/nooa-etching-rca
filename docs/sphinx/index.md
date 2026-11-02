# 刻蚀根因研究系统 · 需求与设计

:::{container} hero
**NOOA Etching RCA**

同一配方，三个业务时段；让研究 agent 从高维传感器与原始曲线中提出可以检查、反驳和修订的异常解释。

本书覆盖从物理仿真、特征生成、CodeActV2 自主实验，到证据门禁、审查、并行研究、实时交付和恢复的完整实现。
:::

## 阅读这份设计的方式

**设备与算法人员**可从[需求](requirements.md)、[数据与物理模型](simulation.md)、[算法设计](algorithms.md)和[实际研究复盘](case-study.md)开始。**开发人员**可先读[系统架构](architecture.md)、[agent设计](agents.md)、[数据契约](data-contracts.md)与[复现指南](operations.md)。**审查者**可直接使用[需求追踪矩阵](traceability.md)和[文档自审](quality.md)，沿实现与测试链接核对每个承诺。

本文标记三类状态：**已实现**指代码里存在相应机制；**已验证**必须注明具体测试或运行；**待改进**不写成现成能力。两者尤其不能混淆：保存表与引用相符，不等于任意研究算法都正确；审查通过，不等于物理根因已经证实。

```{mermaid} diagrams/overview.mmd
:caption: 从固定配方数据到带证据交接的完整链路。虚线表示反馈，不表示已确认因果。
:alt: 固定配方仿真生成公开发现数据，主研究与子分支自由计算，经证据校验和审查后流式交付，专家反馈进入后续研究。
```

## 当前基线

| 项目 | 当前实现 |
|---|---|
| 数据 | 通用ICP硅刻蚀降阶仿真；200通道、11步骤、每次356秒 |
| 研究 | NOOA CodeActV2，外层/科学长驻内核；活对象只在各自当前生命周期 |
| 编辑与实验 | 精确revision编辑/执行，独立冻结输入实验，检查后显式发布，旧结果stale |
| 纠错 | 宿主父子执行证据、原失败路径复验、独立具名目标检查与只读反思 |
| 公式比较 | 任意原始序列算子、版本、固定lot验证与模型比较、最佳重跑及强制交接 |
| 可靠性 | 前后置校验、LoopGuard、Predict审查、最多两次定向修订 |
| 并行 | 主研究动态派生独立分支；默认两个分支活跃 |
| 预算 | 每轮主/子研究、摘要、审查共用原生AdmissionBroker及token账本 |
| 前端 | Vue 3、TypeScript、Vite；持久SSE、Markdown、PNG与CSV表格 |
| 平台 | macOS sandbox-exec；Linux优先Landlock/bubblewrap与seccomp，auto可降级process，严格模式拒绝降级 |

NOOA包版本为 `0.0.11.dev473`，依赖固定在 `bff0ef449f9bd74ddbb19486489f95c0edc73831`，没有修改上游vendor。研究运行配置为272000总上下文、50000输出上限，自动摘要阈值177600。实际值可在[自动生成的配置参考](_generated/contracts.md)检查。

:::{important}
GitHub Pages发布的是静态设计文档，不是研究服务。仓库不包含API密钥、原始私有对话、已有运行数据库或生成的数据集。公开源码包含仿真方法与故障设定；要进行盲评，研究agent仍只能获得导出的公开发现集，不能获得整个仓库。
:::

```{toctree}
:maxdepth: 2
:caption: 需求与系统

requirements
architecture
data-contracts
dynamic-datasets
simulation
```

```{toctree}
:maxdepth: 2
:caption: Agent与研究方法

agents
autoresearch-core
execution-feedback
code-buffers
algorithms
formula-research
feature-isolation
evidence-review
concurrency
streaming
```

```{toctree}
:maxdepth: 2
:caption: 工程与验收

security
repl-runtime
linux-arm
audit-remediation
operations
validation
case-study
decisions
traceability
quality
references
```

```{toctree}
:maxdepth: 1
:caption: 源码生成的参考

_generated/contracts
_generated/tools
_generated/modules
_generated/cli
```
