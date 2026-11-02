# 系统架构与执行边界

## 系统上下文

研究服务运行在持有模型凭据的本地主机。浏览器使用HTTP发消息，通过SSE观察后台任务。FastAPI服务管理会话入口、请求去重和worker生命周期，不直接承载长时间的科学计算。每次研究轮次由独立 `web_worker` 执行；CLI可直接进入同一 `research_turn`。

```{mermaid} diagrams/context.mmd
:caption: 运行拓扑。分支各有自己的研究host、科学worker与NOOA存储；模型请求共用本轮Broker。
:alt: Vue经FastAPI连接独立研究host，主agent协调独立分支，共用预算和根事件数据库，科学worker只读公开输入并写自身工作区。
```

浏览器、HTTP服务和研究host具有不同生命周期。浏览器断开只停止观察；不会取消研究。正常关闭HTTP服务会取消活跃任务并清理后代进程。仅HTTP服务崩溃而研究host存活时，新服务验证进程身份后接管观察；不能把一次页面刷新理解为重启整个agent。

## 分层与模块职责

```{mermaid} diagrams/layers.mmd
:caption: 应用分层及依赖方向。数据生成器与自主研究是独立路径，后者不需要获得生成器真值。
:alt: 界面、HTTP服务、研究编排、agent契约、NOOA原生框架、工作区和审计层的依赖关系。
```

| 层 | 主要模块 | 责任 |
|---|---|---|
| 展示 | `frontend/src/App.vue`, `useResearch.ts` | 会话和方向切换、消息、工具、产物、预算与审查状态 |
| HTTP | `web.py`, `web_worker.py` | 目录限定、消息路由、SSE、任务启动/停止/接管 |
| 编排 | `autoresearch.py`, `research_branches.py` | 本轮对象组合、上下文恢复、分支生命周期、最终发布 |
| 研究 | `autoresearch_agent.py`, `strategy_factory.py` | 公开研究方法、工具、V2/审查/反思策略 |
| 证据 | `research_models.py`, `evidence.py`, `review_pipeline.py` | 输出契约、确定性检查、审查与不可变修订 |
| 执行 | `secure_codeact.py`, `research_workspace.py`, `persistent_kernel.py`, `session_cell.py` | 无凭据外层/科学长驻内核、精确编译、失败失效和输出采集 |
| 源码与实验 | `code_buffers.py`, `code_experiments.py` | 精确revision、冻结输入实验、检查与显式产物发布 |
| 执行观察 | `execution_observations.py`, `execution_provenance.py` | 父子观察、输入/源码/产物来源、原路径复验和目标检查 |
| 请求 | `runtime.py`, `budget.py`, `provider_stream.py` | 模型配置、原生准入、token账本、真实流观察 |
| 数据 | `etch_recipe.py`, `etch_model.py`, `etch_chemistry.py`, `etch_features.py` | 固定工艺、降阶动力学、原始观测及派生特征 |

NOOA提供CodeActV2、Predict/Reflexion、Context/TextSkill、SQLiteStorageManager、QueueManager及AdmissionBroker。项目补的是设备场景知识、证据schema、进程和文件边界、跨进程工具与执行证据桥接、UI身份与交付语义。没有另写一套通用agent运行时。

## 一轮任务的数据与控制流

入口先核对会话，建立本轮共享预算、公开工具和独立工作区；恢复NOOA快照后重新安装运行态对象、技能与上下文。模型生成的外层 `python_cell` 调用应用公开方法。数据查看和连续计算可用 `run_code` / `code_run`；版本候选和公式试错可用 `code_experiment`，保存的脚本可用 `write_file` + `run_script`。科学环境使用只读 `DATASET`、`DATASET_ID` 与各自可写目录；切换数据范围会重建主科学内核，旧变量不再复用。实验先在暂存目录检查，再由宿主显式发布。

候选返回后，应用先核验并冻结证据，再调用原生Predict审查。需要时进入 `revise`，保留每次旧结果。最终结构化答复、审查结果和Markdown先保存，之后原样分段推送规范Markdown。纯可视化通过文件/数值检查后直接交付，不强制科学审查。正文推送失败时保留规范报告及交付状态，不再调用模型重写。

## 两个不同的代码环境

外层NOOA执行环境负责调用公开工具与调度；它的cwd不等于研究workspace。在外层直接 `open('artifacts/...')` 或 `savefig` 可能写到错误位置。科学worker是持久Python进程，cwd才是本方向workspace，可读取本方向选中的不可变数据版本，且不继承API凭据。

同一生成调用的外层 cell 保留自己的 namespace 和事件循环；主科学 `run_code` / `code_run` 则在当前研究轮的另一个内核中复用活对象。二者不直接共享Python对象。编译失败保留原内存，运行异常、超时、取消、崩溃或清理失败使受影响内核失效。新请求和数据切换也不恢复旧活对象。

`run_script` 使用独立临时内核；`code_experiment` 从显式源码revision、JSON参数与冻结输入建立另一个可丢弃worker。其失败不清空主科学内存，也不能覆盖正式产物。源码版本、执行来源、父子观察、修复关系跨轮保留，继续计算必须显式重建所需输入，不隐式重放副作用。

## 状态为什么分开保存

`conversation.sqlite`保存NOOA事件和快照；根 `ui.sqlite`保存浏览器可重放事件、inbox与请求ID；`branches.sqlite`保存任务注册与生命周期；`budget.sqlite`保存本轮token预留和结算；`.datasets/registry.sqlite`保存不可变数据定义和方向独立的选择栈；`.formula-lab/registry.sqlite`保存公式版本及试验。此外，`.execution/code.sqlite` 保存源码revision，`ledger.sqlite` 记录科学来源，`experiments.sqlite` 保存实验manifest，`observations.sqlite` 保存跨层观察及修复关系。它们是不同职责的持久层，不是一个跨数据库原子事务。

因此系统承诺可恢复文件与可追溯状态，**不承诺强杀时所有副作用exactly-once**。进程身份核对、mission匹配、旧报告过滤和显式恢复用于降低误关联风险；运行日志仍需保留来解释崩溃边界。

完整符号与实际公开方法见[源码索引](_generated/modules.md)。

## 数据范围与科学协议

[动态数据集](dynamic-datasets.md)把现场生成的划分代码、父版本、schema、理由和输出哈希绑定为DatasetRef。主/子方向各自选择范围；分支只能使用派生时分配的根及后代。公式计划和数值证据显式绑定版本，避免返回主集后错用旧统计。文件对象和修订快照采用内容寻址，工作区改写不影响已冻结文件。

[特征隔离](feature-isolation.md)区分逐对象算子与学得的变换，fold训练与开发数据走独立入口。外部拉数仅保留类型化可注入接口，未配置时明确不可用；本次没有连接知识图谱或实际外部数据源。

## 审核后的执行与交付路径

可信主进程仅持有NOOA编排、模型客户端、预算和明确工具实现。模型生成的外层CodeAct经JSON代理进入本次生成调用的无凭据长驻进程；科学Python进入当前研究轮的另一个长驻内核，脚本和独立实验例外。二者均不继承live agent。宿主核对完成协议、实际退出状态、来源账本和父子观察，不能仅因进程持续或返回零就证明科学正确。只读Predict reviewer可使用另一已配置模型，全部请求仍受同一预算约束。最终Markdown由核验结果确定性渲染与推送。留出test/仿真参考仅由单独收尾入口评分。细节与测试映射见[审核问题修复](audit-remediation.md)。

## 持久化服务与表结构

项目自有的持久化实现集中在 `autoresearch_agent_v2/database/`，复用 SQLAlchemy 2.0 Core。
数据文件仍按根会话和各研究方向分开，不合并数据库、不改名、不搬动已有记录。

| 部分 | 位置 | 职责 |
|---|---|---|
| 表注册 | `tables.py`、`constants.py` | 用 SQLAlchemy `Table` 和 `MetaData` 声明表、列、主键、默认值及文件分组；导入时不打开数据库 |
| SQLite 实现 | `sqlite/` | 管理连接、短事务、WAL、增量补列及固定业务查询；封装 JSON 编解码 |
| 服务接口 | `services.py`、包入口 | 声明业务方法并创建服务；参数和结果只有业务标识、普通文档与计数 |

上层通过 `create_event_service(root)`、`create_dataset_service(root)` 等入口获取服务，
不导入 SQLite 实现、表对象或 SQLAlchemy，不访问连接，也不提供 SQL 或查询表达式。
`LiveEvents` 已移除公开的 `db`；Web 请求去重、进度水位和审查事件均调用明确的服务方法。
代码版本、数据集发布、公式研究的事务上下文交付受当前事务约束的业务服务，
上层通过其 `get`、`save`、`publish` 等方法完成复合操作，不能透传 SQL。
业务异常保留原有语义；数据库驱动异常统一包装为 `StorageError`。

### 原有表与业务归属

下列为本项目拥有的 10 个文件、17 张表。`TEXT` 中的 `doc`、`body`、`data` 等字段继续保存
原有 JSON 文本，不把历史文档迁移成新模型。除表中写明的复合键外，`id`、`owner`、`key`
或 `request_id` 为相应表的主键。原设计没有数据库级外键，跨文件的身份与来源检查仍由业务层负责。

| 文件（相对会话目录） | 表 | 列与业务 |
|---|---|---|
| 根 `ui.sqlite` | `events` | `id INTEGER` 自增；`task_id/type/data TEXT`、`created REAL`、`agent_id/parent_agent_id TEXT`；供 SSE 回放、方向进度和反思审查 |
| 根 `ui.sqlite` | `inbox` | `id INTEGER` 自增；`task_id/message TEXT`、`consumed INTEGER` 默认 0、`agent_id/source TEXT`；按任务及方向消费指导 |
| 根 `ui.sqlite` | `requests` | `request_id/task_id TEXT`；将重试请求绑定到首次登记的任务 |
| 根 `.datasets/registry.sqlite` | `datasets` | `id/doc TEXT`、`created REAL`；不可变版本文档、内容身份、版本族及血缘 |
| 根 `.datasets/registry.sqlite` | `scopes` | `owner/root_id/current_id/stack TEXT`；方向独立的初始版本、当前版本和选择栈 |
| 根 `.datasets/registry.sqlite` | `meta` | `key/value TEXT`；记录会话初始数据集 |
| 根 `.datasets/registry.sqlite` | `acquisition` | `owner/request_id/request/result TEXT`，复合主键 `(owner, request_id)`；保存采集参数和结果，隔离方向内的请求身份 |
| `.formula-lab/registry.sqlite` | `studies` | `id/body TEXT`；公式协议、源码版本、试验、选优和重放历史 |
| 根 `branches.sqlite` | `branches` | `id/parent_id/mission/title/task/state TEXT`，`created/updated REAL`，`pid INTEGER`，`process_created REAL`，`review_status/error TEXT`；方向身份和生命周期 |
| 根 `budget.sqlite` | `tokens` | `id/mission/agent/purpose TEXT`，`reserved/charged INTEGER`，`state/usage TEXT`，`created REAL`；共享预算预留及结算 |
| `.execution/code.sqlite` | `versions` | `name TEXT`、`revision INTEGER`、`source/sha256 TEXT`，复合主键 `(name, revision)`；只追加源码版本 |
| `.execution/ledger.sqlite` | `sequence` | `value INTEGER NOT NULL`，沿用原来的单行无主键结构；分配执行编号 |
| `.execution/ledger.sqlite` | `cells` | `id/doc TEXT`；只追加输入、源码和产物哈希对应的可信执行文档 |
| `.execution/experiments.sqlite` | `experiments` | `id/doc TEXT`；实验冻结输入、依赖、检查和发布 manifest |
| `.execution/observations.sqlite` | `observations` | `id/doc TEXT`；宿主记录的执行观察、父子关系和检查结果 |
| `.execution/observations.sqlite` | `repairs` | `failure/verification TEXT`，复合主键 `(failure, verification)`；失败观察与原路径复验关系 |
| `.evidence/proofs.sqlite` | `proofs` | `id/doc TEXT`；可重算证据及其输入、结果和引用绑定 |

`conversation.sqlite` 由 NOOA 的 `SQLiteStorageManager` 创建和管理，不纳入这 17 张自有表。
审计服务使用只读连接读取 NOOA 的事件计数、Summary 和活跃标签；不会创建缺失文件或迁移第三方表。

### 事务与旧库兼容

所有写事务在读取共享状态前取得 `BEGIN IMMEDIATE` 写锁，异常时回滚并关闭连接。
预算检查与预留、代码版本冲突检查与追加、执行编号分配、指导消息消费、请求身份登记，
分别在单个事务内完成。数据集版本号与发布文档在同一业务事务中写入；文件发布仍遵循
原有内容寻址流程，不承诺数据库与文件系统之间的原子提交。

表注册使用 `MetaData.create_all()` 创建缺失表。旧 UI 库缺少的 `agent_id`、
`parent_agent_id` 和 `source` 在写锁保护下增量补齐，旧记录分别使用 `main`、空父方向及 `user` 默认值。
已有记录、执行编号和代码 revision 保留。首次打开新文件时并发切换 WAL 的锁冲突只在连接配置阶段
有限等待重试，不自动重放业务事务。

`tests/test_database_services.py` 验证旧库读取与补列、回滚、方向隔离、只追加记录、只读审计，
并使用独立进程竞争预算、代码版本、指导消息、请求身份和执行编号。
同一测试文件检查应用层不导入数据库内部实现、不嵌入 SQL，防止新增调用绕过服务接口。
