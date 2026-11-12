# V2 通用多 Agent 架构改造设计

本文依据 2026-11-12 的工作区源码编写。当前分支为 `main`。本版纳入工具分层、allow_tools 与并行子 agent。另补充实时结果投递和通用 finish 路由。前稿中的专用 ResearchRunner 与固定研究合同设计已撤回。

本设计的通用框架已实施：工具注册与授权、skill 加载、并行子会话、finish 路由、可靠投递和通用 JSON 文档均已接入。旧 `feature_verifier/` 已删除，纯数值评估移至历史模块。业务算法与前端未改写。

本文保留改造前的源码定位与实施计划，作为设计依据。当前可用接口与运行说明见 [Autoresearch 核心 Agent](autoresearch-core.md)。框架不附带默认 Auto Research 业务 skill。

## 架构边界

所有 agent 使用同一个 Session、AgentLoop 和基础 system prompt。主子关系由运行时身份表示。研究者、复核者或规划者的分工来自 skill、用户请求与父任务，无需独立的 Agent 实现类或角色专用 prompt。

通用框架负责工具注册与授权、skill 加载、独立会话、并行调度、消息投递、结束路由、持久化和资源隔离。业务负责研究目标、合同字段、结果格式、科学验证标准和研究 JSON 的语义。

| 事项 | 通用框架 | skill、用户请求与业务扩展 |
|---|---|---|
| Agent 类型 | 一套 Session / AgentLoop | 通过任务和 skill 形成分工 |
| 任务传递 | 通用文本与附件引用 | 目标、方法、完成标准、输出模板 |
| 结果传递 | finish 内容、引用与宿主身份 | 指标、结论、证据和领域字段 |
| 工具 | 注册、过滤、调用、资源生命周期 | 工具实现与使用策略 |
| Skill | 发现、加载、版本与引用读取 | 研究组织、执行、审查和领域知识 |
| 并行 | 并发上限、排队、预算、取消与恢复 | 如何拆分任务、是否继续研究 |
| 状态 | 会话、委派、执行、收件箱 | 研究验收、最佳方案、假设和反证 |
| JSON | 通用文档存储、版本和写入权限 | research.json 的字段与维护时机 |

基础 system prompt 只规定通用行为与证据诚实。它不固定研究轮次、五折验证、任务合同、研究 JSON 字段或特定验收流程。工具说明仍需声明真实参数和通用调用规则。

```{mermaid} diagrams/generic-agent-runtime.mmd
```

## 改造前实现与代码定位

| 位置 | 当前事实 | 需要调整 |
|---|---|---|
| `agent/prompts.py:32` | 默认 prompt 包含研究轮次、Task、Verifier 和固定交付职责 | 提取通用基础规则；业务流程移交 skill |
| `agent/typing.py:42` | 已有 agent_id、parent_agent_id 和独立 messages | 保留；补充通用委派与收件箱状态 |
| `agent/loop.py:63` | 拼接基础 prompt、数据位置和所有工具说明 | 使用统一上下文构建器，只展示已授权工具 |
| `agent/loop.py:79` | 同一 Session 跨 run 累积消息 | 新委派创建新 Session，避免继承父完整历史 |
| `agent/loop.py:174` | finish 直接返回循环结果 | 交给运行时按父子身份路由 |
| `agent/session.py:199` | 启动时绑定所有已登记资源和工具组 | 先解析 allow_tools，再绑定获准资源 |
| `tools/base.py:49` | ToolFactory 管理工具与 Schema，尚无统一授权策略 | 分离注册目录、会话选择和调用执行 |
| `tools/builtin_tools/messages.py:4` | finish 打印结果，语义指向用户 | 结果发送与工具函数解耦 |
| `python_tools/base.py:18` | PythonTool 使用进程级注册字典 | 显式归属 Registry，避免跨运行装配泄漏 |
| `python_tools/runtime.py:41` | 会话复制全部已登记 Python 类 | 复制授权后的类与方法快照 |
| `python_tools/registry.py:5` | 自动导入 statistics 作为内置工具 | 内核不自动导入业务包 |
| `feature_verifier/agent.py:78` | 专用 verifier、持续子会话与串行队列 | 删除该专用 agent，使用通用并行运行时 |
| `feature_verifier/prompts.py:23` | 独立 VERIFIER_PROMPT | 删除；任务差异由 skill 与父任务表达 |
| `tools/builtin_tools/tasks/tools.py:20` | Task 只是工作项 CRUD | 保留为可选通用清单，不承担子会话调度 |
| `agent/history.py:62` | 原始历史保留，模型请求使用压缩投影 | 保留；补充收件箱去重与 skill 投影 |

当前 Task 列表只返回少量摘要，主要上下文增长来自主 agent 的研究执行轨迹及反馈。现有 verifier 也积累历轮上下文。该判断来自静态调用链分析，未测量实际 token 分布。

## 目录与依赖方向

下图说明职责与依赖方向。实际运行时将调度、消息与结束路由集中在 `runtime/manager.py`，以存储和工厂分隔状态与装配，不为每个概念创建单独模块。

```text
src/factor_demo/
  autoresearch_agent_v2/
    agent/                         共用 Session、Loop、基础 prompt
    runtime/
      manager.py                   AgentRuntime 与会话树
      scheduler.py                 并发、排队、预算与取消
      mailbox.py                   父会话收件箱
      routing.py                   finish 与唤醒路由
      store.py                     执行、投递与恢复记录
      typing.py                    通用委派、结果和事件类型
    tools/
      registry.py                  工具定义与 provider 注册
      policy.py                    allow_tools 解析与继承
      base.py                      会话工具执行入口
      builtin_tools/
        messages.py                finish
        agents/                    spawn、status、wait、cancel
        skills/                    list、load、read_resource
        records/                   通用 JSON 文档访问
        files/                     通用文件工具
        notebooks/                 Notebook 执行
        sandbox.py                 隔离 Shell
        tasks/                     可选工作清单
    skills/                        文本技能发现、加载与快照
    python_tools/                  Python 类接口适配与 Notebook 桥接
    sandboxing/                    保留内部执行隔离
    database/                      保留公共存储基础设施
  business_tools/
    __init__.py                    显式业务注册入口
    ...                            独立业务工具包，本轮不设计具体内容
  knowledge/                       skill 正文及参考资料
```

依赖方向是业务工具实现框架协议，应用启动时将业务工具注册到框架。通用内核不导入 business_tools，也不扫描文件夹执行任意模块。业务工具放在内核包外，避免通用运行时与具体研究能力形成反向依赖。

普通可调用工具和 Notebook 中的 Python 类工具都进入统一工具目录。PythonTools 保留为适配层，复用已有类接口检查、按需实例化和隔离桥接。

`feature_verifier/` 整个目录列入删除计划。不新增 research_agent/、review_agent/ 或角色专属 Agent 类。独立上下文由通用 Session 实例提供。

## 工具注册与 allow_tools

### 注册、启用与实例化

工具分为 builtin 与 business。builtin 提供通用运行能力；business 由独立目录提供并显式注册。两者共用 ToolRegistry、调用上下文与授权检查。

ToolSpec 建议包含稳定 ID、公开调用名、来源类型、provider、说明、参数定义或静态接口描述，以及会话绑定工厂。来源类型只表达归属，不自动授予额外权限。

注册只登记定义和工厂，不创建业务资源。会话启动时依次执行：登记可选工具、解析 allow_tools、检查依赖、绑定实际需要的资源、生成授权后的工具表。全部成功后冻结会话工具集。

业务注册可使用显式函数或装饰器，但注册目标必须是应用持有的 Registry 实例。多个运行实例不能共享可变的默认全局工具表。

以下是拟新增接口的示意。名称用于说明装配关系。

```python
registry = ToolRegistry()
register_builtin_tools(registry)
register_business_tools(registry)

parent = Session(
    model=model,
    state=state,
    tool_registry=registry,
    allow_tools="all",
    skills=["autoresearch"],
)
```

### 选择规则

`Session.allow_tools` 接受 `"all"` 或工具 ID 列表，默认 `"all"`。all 表示本次应用已注册、已启用且获得宿主授权的工具全集。未注册工具不会因 all 自动加载。

列表支持精确工具 ID。可增加显式 provider 选择形式，如 `business:Statistics`，展开为该 provider 的公开方法；第一版不引入复杂通配符规则。未知 ID 在启动或委派时直接报错，避免静默配置错误。

```text
应用可用工具 = 已注册 ∩ 本次应用启用 ∩ 宿主授权
当前会话工具 = 应用可用工具 ∩ allow_tools
子会话工具   = 当前父会话工具 ∩ 子 allow_tools
```

子 agent 的 allow_tools 同样默认 all，含义是使用父 agent 可委派的当前工具全集。父 agent 可以选择子集，不能通过子会话扩大自身能力。文件、网络和凭据权限继续由独立的宿主资源授权限制。

finish 也遵守工具选择。当前循环依赖 finish 结束工作，因此有效工具集中缺少 finish 时，启动校验失败。`allow_tools=[]` 在这一运行模式下无效；不会暗中重新加入被排除的工具。后续如提供无工具单次推理模式，应独立声明终止语义。

### 所有调用入口执行同一策略

allow_tools 必须同时作用于以下位置：

- 模型工具 Schema 与工具使用说明。
- Session.call 和 ToolFactory.call 的实际调用入口。
- python__describe_class 返回的目录、类文档与方法清单。
- Notebook 代理方法和宿主 RPC 桥接的每次调用。
- 子会话启动参数、恢复检查与运行时管理接口。

部分授权的 Python 类只公开获准方法。不能通过 describe_class 重新开放未授权方法，也不能在记录过方法名后绕过调用检查。资源工厂只为获准工具及其必要依赖启动。

控制某个业务工具的调用权限，不等于禁止 Notebook 导入同类科学库。若需要限制代码执行，应限制 Notebook/Shell 能力及其沙箱资源，不能把工具名单当作 Python 包隔离机制。

## 共享 prompt、skill 与任务传递

主子 agent 使用同一个 BASE_SYSTEM_PROMPT 和上下文构建器。身份、工具权限、父子关系来自宿主。研究流程由已加载 skill 与任务请求表达。

SkillRegistry 提供技能目录；SkillLoader 提供正文、引用文件、版本和哈希快照。首次只展示简短目录，正文按需加载。父 agent 指定的 skill 在子 agent 第一次请求前直接预加载。

skill 可以声明工具依赖；依赖用于检查，不会修改 allow_tools。缺少必需工具时应在启动前报告错误，或按技能明确声明的可选路径工作。skill 不执行注册代码，也不直接获得宿主凭据。

通用委派工具建议保持最小输入：

```text
agent__spawn(
  task: str,
  skills: list[str] = [],
  allow_tools: "all" | list[str] = "all"
) -> {agent_id, assignment_id, status}
```

可补充通用的上下文引用与较小资源预算。父子身份、会话 ID、写目录与实际权限由宿主生成，模型不能自行指定。

`task` 是父 agent 提供的任务正文，可以是自然语言、Markdown，或由 skill 约定的 JSON 文本。框架只检查文本大小、引用边界及传输完整性。

框架不要求 task 包含 objective、acceptance_criteria、research_id、protocol_ref 或特定结果字段。它们可以由 Auto Research skill 或用户请求约定，但不是 system prompt 或内核 Schema 的固定要求。

同理，finish 不强制返回 ResearchResult。需要结构化输出时，skill 或用户请求约定正文格式与附件。可选业务工具能校验特定格式，但不能成为所有 Agent 的前置条件。

默认子 agent 从空历史开始，只接收所分配任务、所选 skill 和必要引用。不同子任务隔离历史；同一委派的继续执行可恢复原会话。恢复时保留 skill 快照与原授权上限；当前宿主授权收紧后不得恢复已撤销能力。

切换 skill 时重建激活内容投影，不重复追加完整正文。旧历史继续保留。若原消息会重新引入失效的角色规则，应新建会话并传递事实摘要。

## 多子 agent 并行

新增 AgentRuntime 管理会话树，Scheduler 管理独立执行任务。主子 agent 都是相同的 Session。单个 Session 内继续串行维护模型历史，不在同一 messages 上并发运行两个循环。

agent__spawn 立即返回句柄。运行时在并发槽位可用时启动子会话，超限时持久化排队。多个已启动子会话同时执行，不等待前一个 finish 后再投递下一份任务。

并发上限作为框架配置，不写入角色 prompt。首个可用版本必须通过两个以上子会话实际重叠运行的检查，不能以串行版本作为最终交付。上限、总预算和递归深度由宿主设置；父 agent 可在上限内收紧。

子 agent 能否继续派生，取决于它是否获准使用 agent__spawn，以及剩余深度与预算。不按“research agent”名称写死权限。

agent__status、agent__wait 和 agent__cancel 只能管理调用者获准访问的委派。不能通过其他 agent 的 ID 读取无关任务、取消父会话或扩大管理范围。

每个子会话独占写目录、Notebook kernel、工具实例与模型历史。只读输入和冻结结果按授权共享。全局运行时只在更新注册表或状态时短暂加锁，不能在模型调用或工具执行期间持有总锁。

父 agent 等待结果时进入挂起状态，释放模型执行资源。子结果到达后唤醒。agent__wait 用于父 agent 主动等待，不是自动接收结果的必要条件。保留父会话继续处理结果所需的调度机会，避免子任务占满全局资源后父会话无法收尾。

取消沿所属会话树传播，排队委派标记取消，运行中的调用和 kernel 执行清理。失败、超时与取消以运行事件返回父会话，不伪装成正常 finish。

## finish 的结果流向

finish 的唯一通用含义是结束当前分配的工作，并将内容交给运行时。工具不决定研究是否通过，也不在函数内部直接面向用户打印交付。

| 调用者 | 运行时路由 | 影响 |
|---|---|---|
| 根 agent | 根输出通道 → 用户 | 结束当前用户请求 |
| 子 agent | 父会话收件箱 | 结束当前子委派并唤醒父会话 |
| 孙 agent | 直接父会话收件箱 | 沿树回传，不越级发送 |

子 agent 完成后，运行时冻结 finish 内容，生成稳定 result_id，并持久化结果和投递事件。随后将事件送入父收件箱。子 agent 无需调用独立 report，也不需要先经过固定 research_submit_result 协议。

模型提供的 content 是结果材料。发送方、接收方、assignment_id、result_id 和内容哈希由宿主补充，正文不能伪造这些身份或改变流向。

结果过长时保存全文，向父会话投递有界摘要和 result_ref。原始结果保持完整，可按需读取。为保证通用性，摘要不依赖固定研究字段；正文长度及预览预算由运行配置决定。

### 实时投递的安全边界

子结果提交后立即进入父收件箱，不等主 agent 主动查询。父会话的消费时机如下：

| 父会话状态 | 处理方式 |
|---|---|
| 空闲或主动等待子任务 | 唤醒原 Session，从下一次模型请求继续 |
| 正在准备模型请求 | 收取已到达事件，然后构建请求 |
| 正在等待模型响应 | 事件立即排入收件箱；请求返回后处理 |
| 正在执行工具批次 | 批次完整回执后插入事件，再请求模型 |
| 正准备 finish | 先检查收件箱与子会话状态，再提交终止决定 |

已经发送的模型请求不能在中途加入新上下文。默认不取消这次请求，也不把子结果插入尚未完成的 assistant/tool 消息批次。父会话的最终回复不能越过已到达但尚未处理的结果。

持久化的收件箱与唤醒信号独立于 Session 的长运行锁。外部子会话只提交事件，不直接修改父 messages。父循环是历史唯一写入者，在安全边界将事件转换为来源明确的消息。投递使用独立外部事件消息，不伪造 tool_call_id，也不将子结果提升为 system 指令。

同一父会话只保留一个驱动协程。唤醒合并为待处理信号，不能同时调用两次 Session.run。空闲唤醒仅发生在仍归运行时管理的请求中；根请求正式结束后，不自行开启新的用户请求。

多个子结果按到达顺序投递，每项保留稳定 ID 和原始父任务引用，不等待较慢子任务。一次边界收取有数量与长度上限，避免大量并行结果同时填满上下文；剩余事件继续保留并立即安排后续消费。

### 结束竞态与恢复

父 finish 与子结果提交共用短事务或短临界区，避免结果在终止检查后到达却被遗漏。默认结束策略要求子委派已完成或明确取消，且已投递结果已读入父上下文。尚有活动子任务时，finish 返回未结束原因，父循环继续。

框架只检查子任务生命周期和消息是否收取。研究验收是否充分，由 skill 和用户要求控制。用户停止时走取消路径，可保留部分结果，不要求完成研究。

finish 的结果落盘、委派终态和 outbox 记录在同一事务提交；消息投递允许重试。模型历史检查点保存已消费 event_id，恢复时依据 event_id 去重。崩溃发生在检查点保存与确认之间时，不重复插入同一个逻辑结果。

最终完成与外部输出分别保存状态。先持久化终止决定再发送根结果；输出通道如支持幂等键则使用 result_id。不能承诺对所有外部副作用实现 exactly-once。

同一批工具中的 finish 之后的调用继续按现有协议返回未执行回执。若有待收取结果导致 finish 未通过终止检查，仍须补齐该批次回执，再收取事件并继续。

## 持久化、JSON 与隔离

通用运行存储负责 agent 树、委派正文、执行状态、skill 快照、工具授权、结果引用和收件箱。建议复用现有 SQLite 事务基础设施，新增运行时存储服务，不引入研究领域表。

研究 JSON 继续由主 agent 维护。建议提供通用 records 工具操作宿主管理的 JSON 文档，支持 document_id、JSON 内容或补丁、expected_revision，以及按文档设置的所有者权限。工具校验 JSON 合法性和并发版本，不要求固定研究字段。

research.json 的结构、何时更新、如何记录结论与验收，由 Auto Research skill 与用户请求约定。本轮不确定具体业务 Schema，也不新增 research_create、research_review 或 ResearchResult 等固定业务合同。

研究文档不能成为运行队列或投递可靠性的唯一来源。运行时数据库记录真实执行状态；研究 JSON 记录主 agent 的研究认识。二者语义不同，无需将 Task、运行状态和研究验收做成多个互相覆盖的权威来源。

记录与会话检查点放在模型可写目录之外。父子 agent 对这些资源的访问通过受控工具完成。子会话默认不能改写父文档；授权按资源所有者与工具范围执行，无需定义专用角色类。

执行保持宿主原生 Python、uv、Node，并保留内部沙箱。各会话独占写目录，显式授予只读输入，不继承凭据或越界路径。缺少受支持隔离能力时报告阻塞，不改用外层 Docker，不取消隔离。

Skill 资源加载拒绝越界与符号链接逃逸。工具过滤不能代替执行沙箱；共享基础 prompt 也不能代替调用入口的授权校验。

## 文件级修改与删除计划

下表是实施所依据的修改清单。相对路径均位于 `src/factor_demo/autoresearch_agent_v2/`，除非注明完整仓库相对路径。

| 类型 | 文件 | 框架改动 |
|---|---|---|
| 修改 | `agent/prompts.py` | 通用基础 prompt；迁出研究轮次、Verifier 和固定合同 |
| 修改 | `agent/typing.py` | 记录通用委派、skill 快照、有效工具引用和收件箱消费位置 |
| 修改 | `agent/session.py` | allow_tools 默认 all；注册后过滤；按需资源绑定；支持运行时挂起与恢复 |
| 修改 | `agent/loop.py` | 消费 inbox、保证工具批次完整、统一 finish 交接与终止检查 |
| 修改 | `agent/history.py` | 压缩保留结果身份与未处理工作；避免重复插入 skill 或事件 |
| 新增 | `runtime/{manager,scheduler,mailbox,routing,store,typing}.py` | 通用会话树、并行、投递、路由、恢复和取消 |
| 修改 | `tools/base.py` | 工具执行与授权；按工具归属过滤说明；移除直接输出依赖 |
| 新增 | `tools/{registry,policy}.py` | 显式 provider 注册；工具来源、名单与继承规则 |
| 新增 | `tools/builtin_tools/agents/` | 通用 agent__spawn、agent__status、agent__wait、agent__cancel |
| 新增 | `tools/builtin_tools/skills/` | 技能目录、加载与引用读取 |
| 新增 | `tools/builtin_tools/records/` | 通用 JSON 文档读写、版本冲突和所有者权限 |
| 修改 | `tools/builtin_tools/messages.py` | finish 只提交通用完成信号，发送目的地由 runtime 决定 |
| 修改 | `tools/__init__.py` | 导出注册接口与内置工具，避免导入业务包 |
| 新增 | `skills/` | 已批准目录发现、预加载、正文与引用快照 |
| 修改 | `python_tools/{base,registry,runtime}.py` | 接入实例 Registry，过滤类和方法；桥接调用再次授权 |
| 调整归属 | `python_tools/builtin_tools/statistics.py` | 统计工具退出内核自动导入；移动至独立 business_tools 并显式注册 |
| 复用 | `sandboxing/`、`tools/builtin_tools/notebooks/` | 每会话隔离资源，保证并发取消与清理 |
| 删除 | `feature_verifier/agent.py` | 删除 FeatureAnalyzeVerifier 与专用 VerifierModel |
| 删除 | `feature_verifier/prompts.py`、`feature_verifier/__init__.py` | 删除专用角色 prompt、导出和 report 协议 |
| 移出后删除目录 | `feature_verifier/evaluation.py` | 如需保留纯评估功能，迁往业务工具或历史目录；内核不依赖该算法 |
| 扩展 | `config.py` | allow_tools、并发上限、委派深度、预算与 inbox 注入限制 |
| 扩展 | `database/services.py` 与相应 SQLite 实现 | 通用运行状态、事务投递和文档访问；保留其他模块使用的既有服务 |
| 新增目录 | `src/factor_demo/business_tools/` | 定义与内核协议对接的独立注册边界，不展开具体业务实现 |
| 文档 | `docs/sphinx/autoresearch-core.md` 与架构图 | 替换持续 verifier、共享写目录和直接输出的旧契约 |

旧导入方已清理：`src/factor_demo/autoresearch_web.py`、`scripts/experiments/run_sensor_similarity.py` 和 `scripts/experiments/run_notebook_only_case03.py`。旧 verifier 生命周期测试由通用运行时测试替代，纯数值回归保留在 `tests/test_feature_evaluation.py`。

这些入口后续只做必要的通用运行时装配与依赖清理。Web 业务流程、前端展示、实验问题和验证算法的重设计均不在本轮范围。旧运行文件可保留为历史资料，不恢复已经删除的专用 Agent 类。

不新增 role=research 对应的类或 system prompt 分支。角色名称可作为展示文本，不能成为代码选择执行路径的依据。

## 实施阶段与验收

| 阶段 | 改动 | 验收标准 |
|---|---|---|
| 1 | 注册目录、builtin/business 分离与 allow_tools | all 只开放已注册能力；精确选择、未知工具、跨会话隔离及 Python 桥接均通过 |
| 2 | 共用基础 prompt、skill 与通用委派 | 父选技能首次请求生效；任意任务正文可传递；没有固定研究合同 |
| 3 | AgentRuntime、Scheduler 与独立 Session | 至少两个子 agent 重叠执行；单会话历史串行；超过上限排队 |
| 4 | finish 路由、实时 inbox 与恢复 | 子完成自动唤醒父；工具协议完整；投递去重；结束竞态不丢结果 |
| 5 | 通用 JSON records 与状态恢复 | 版本冲突、所有者隔离与恢复可检查；不强制研究字段 |
| 6 | 删除 feature_verifier、清理导入与更新说明 | 核心无专用角色类；现有入口通过最小装配接入通用运行时 |

建议新增 `tests/test_tool_registry.py`、`test_tool_policy.py`、`test_v2_skills.py`、`test_agent_runtime.py`、`test_agent_mailbox.py`、`test_finish_routing.py` 与 `test_record_tools.py`。复用现有 Session、history、Notebook、沙箱和 PythonTools 测试。

核心并发测试采用可控的模拟模型和实际异步调度。两个子会话都进入等待点后再同时释放，证明存在执行重叠；不能仅根据耗时接近判断并发。

实时投递测试分别覆盖父会话空闲、等待模型、执行工具、主动等待与准备 finish。检查新事件在下一处安全边界进入模型请求，不需要 status 查询，并验证恢复与重复事件不会造成重复处理。

工具权限测试必须直接调用被拒工具，并尝试从 describe_class、Notebook 代理和宿主桥接绕过。单纯检查模型看不到工具名不足以证明授权有效。

业务质量、科学正确性和研究 token 节省率留给后续应用验证。本轮架构验收使用通用任务，不引入特征评估、五折验证或刻蚀案例作为内核成立的条件。
