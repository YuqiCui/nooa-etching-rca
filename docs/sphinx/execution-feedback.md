# 执行反馈、失败证据与反思上下文

每次科学执行都有宿主归档，即使外层只是 `r = await self.run_code(...)` 而没有打印结果，下一轮也会收到执行观察。工具代理自动附加的状态与动态 Context 互相补充；外层 cell 正常结束不表示所有内层执行都通过，外层随后崩溃也不撤销已发生的内层记录。

## 三类记录和读取入口

| 记录 | 保存位置 | 读取方式 |
|---|---|---|
| 单次科学源码、stdout/stderr、输入/产物来源 | `.execution/cell_*/` 与 `ledger.sqlite` | `list_executions`、`read_execution` |
| 独立实验的精确版本、参数、检查和有效性 | `.execution/experiments.sqlite` 与实验目录 | `code_read_experiment` |
| 外层、科学执行、实验、修复与目标观察的父子关系 | `.execution/observations.sqlite` | `read_observation` 与动态 `execution_evidence` |

`execution_ref` 标识所属工作区的 cell，`experiment_id` 标识独立实验，`observation_id` 标识跨层宿主观察；不能互换。外层异常以文本调用栈跨进程返回，不传递可反序列化的 live exception 对象。复验应引用失败的 observation ID；失败实验及其内部失败 cell 可以是不同记录。

`read_execution(execution_ref, offset=0, limit=12000)` 按字符分别分页源码、stdout、stderr和组合输出，各字段带总字符数和 next_offset。`read_observation` 分页读取包含状态和父 ID 的 JSON 文本。`research_cells/` 中的可写输出只是便利副本，权威读回来自宿主私有归档。

## 有界输出不等于完整无限日志

科学 stdout 保留开头、stderr 保留末尾，各上限 262144 字符，超限后仍继续排空，避免管道阻塞。返回的输出预览同时保留首尾并标记截断。分页能读取的是已经保留的内容，不能恢复采集时被丢弃的中段。源码与输出的长度及分页偏移应分别处理。

科学完成帧与 stdout/stderr 分开并关联请求；打印一段成功 JSON 不能完成 cell。完成帧仍来自不可信 Python 进程，不是对数学或恶意同进程伪造的独立保证。崩溃、协议失败和非零脚本退出仍须核对宿主状态。

## 内存状态和失败清单

科学执行结果分别报告 `kernel_id`、`epoch`、`runtime_status`、`state_effect` 与异常类型，不能只看输出字符串。编译失败保留当前 namespace；运行异常、超时、取消、崩溃或清理失败使受影响内核失效。脚本和独立实验有自己的内核；它们失败不表示主 REPL 被清空。新请求和数据切换不会自动恢复旧活对象。

当前内存字段 `latest_execution/latest_failure` 只是本 `PythonWorkspace` 的快捷信息；持久未解决失败清单来自 observations 数据库，重建宿主后仍可读取。动态上下文给出最近 3 项观察、最多 6 项未解决失败及总数，文本按字段限长并指向完整读取工具。失败的外层 cell 作为编排诊断保留；未解决清单跟踪独立的科学、实验和检查失败。

原生外层的返回值、辅助接口、局部变量描述和异常由NOOA负责；项目通过公开execute_python中间件追加父子观察，不重写ExecutionResult的captured locals或返回值。普通外层异常可能保留部分状态，不能假定它与科学内核的失败重置相同。原生worker硬终止时尚未回传的外层输出可能丢失，但宿主已完成的科学工具记录仍保留。

## 纠错和验收使用同一份证据

下一轮应先读对应源码、checks、异常和输入范围，再编辑新 revision。`verify_repair` 重放原失败路径，并检查输入和当前版本；只有这项显式复验能建立修复关系，无关执行成功不能清掉旧失败。`verify_goal` 另外执行明确的具名目标检查；修复通过的 `goal_satisfied` 仍为 `None`，不能直接当作目标完成。

`inspect_candidate` 以 `execution_evidence` 参数接收同一宿主 Context 的快照。它只能读现有观察和产物，不能为了“反思”执行实验或改代码。确定性模型测试验证实际下一轮请求里存在源码、异常与父子观察；真实 MiniMax 的限定纠错验收另外记录，不能混称为任意研究都能自行纠正。完整语义见[运行协议](repl-runtime.md)。
