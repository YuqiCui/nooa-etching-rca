# Autoresearch 核心 Agent

`autoresearch_agent_v2` 的最小会话状态由
`src/factor_demo/autoresearch_agent_v2/agent/typing.py` 中的 `SessionState` 定义。

## 网页阶段性进展

网页宿主为新版 Agent 挂载 `report(content: str)`，只接收一个非空正文字符串。
`report_progress` 已关闭，当前只提供 `report`。
正文使用 Markdown，可直接包含表格、`artifacts/` 图片引用和文件链接。
该工具通过现有 `reply.commit` 事件发布进展，前端将其显示在所属 Agent 的对话中。
调用后继续执行当前任务，只有 `finish(content)` 提交最终结果并申请结束工作。

研究模式和仅讨论模式均可汇报进展，仅讨论模式的文件及执行权限保持原有约束。
Worker 仅在获准使用对应汇报工具时可调用，消息保留 Worker 身份。
工具说明随实际授权提供，不改变基础 Agent 的默认工具集合；独立脚本需由宿主另行挂载。
消息随现有事件持久化并支持页面恢复，同一会话内相同工具调用标识更新同一条消息。
这项能力在工具执行时发布完整正文，不提供逐字流式生成或长时间代码执行中的自动播报。

## 独立运行传感器特征分析

`scripts/experiments/run_sensor_similarity.py` 使用核心 Agent 执行
`data/sensor-similarity-demo-20261030/task.txt` 中的原任务：挖掘 A、B 两组的
Top 10 相关特征，说明排名依据、组间表现，并保存分析代码及数据或图表证据。
脚本直接使用 Session、文件工具、隔离 Shell 和渐进发现的 Python/Notebook 工具。

```bash
# 只查看计划，不创建运行目录、不读取凭据、不调用模型。
uv run python scripts/experiments/run_sensor_similarity.py --dry-run

# 实际运行；使用 .env 中的 custom provider 和 configs/runtime.json。
uv run --env-file .env python scripts/experiments/run_sensor_similarity.py

# 仅覆盖模型名称，保留同一服务和凭据。
uv run --env-file .env python scripts/experiments/run_sensor_similarity.py --model deepseek-v4.1-flash
```

默认每次创建独立的 `runs/sensor-similarity-v2-<时间>-<标识>/`，
也可用 `--run-dir` 指定尚不存在的目录。整次任务默认沿用运行配置，当前限时 7200 秒，
`--timeout` 可修改此值；`--tool-timeout` 设置单次代码执行上限，默认 120 秒。
工具调用统一使用 LiteLLM 的原生 `message.tool_calls`，不再提供正文 JSON 模式。
脚本沿用运行配置的 thinking 开关、推理强度和请求超时，但不发送 `max_tokens`
或 thinking 的 `budget_tokens`。请求时限和重试策略均沿用运行配置。
DeepSeek thinking 使用兼容的 `tool_choice="auto"`，其他模型默认使用 `required`。
GLM 同旧架构一样显式设置 `thinking.clear_thinking=false`，保留推理历史。

数据复用旧 Agent 的 `discovery_copy` 导入流程：提供样本索引、对应曲线、步骤、
传感器说明、特征说明、配方及数据集声明，排除 test/holdout 样本。
不导入 `features.parquet`、`group_A.parquet`、`group_B.parquet`，也不授权读取
原始 `public/`、生成器、隐藏真值、基线或旧运行结果。宿主在 `input/` 保存筛选后的输入，
准备任务时将其复制到新的 `workspace/dataset-*/` 目录，并将该副本登记为 `DATASET`。
文件工具和执行工具使用同一份工作目录内数据，不再额外授权读取外部的 `input/`。
该副本属于可写工作目录；修改副本不会改变宿主保存的输入。
分析产物写入 `workspace/artifacts/`。续跑也从保存的输入创建新副本。

宿主在工作目录之外保存 `input_manifest.json`、`task.txt`、`session.json`、
`status.json`、不含凭据和服务地址的 `run_config.json`，以及最终的 `final_response.md`。
`model_events.jsonl` 逐次保存完整模型请求、回复及时间；每次请求前更新会话检查点。
每次响应的 LiteLLM 用量原样保存到
`token_usage.jsonl`，累计输入、输出、总 token 及缓存和推理明细写入
`token_summary.json`；缓存和推理明细不再次加到总数中。
缺失或不完整的用量分别计数。失败、超时和取消时仍保存已收到的用量与会话。
用量仅覆盖收到的服务端报告；探针、内部重试以及失败或取消请求中未报告的消耗，
不能从这些记录中完整还原。

失败、超时或取消后，可用 `--resume-from <原运行目录> --run-dir <新目录>` 续跑。
恢复要求模型生成配置及数据集路径匹配，拒绝复制含符号链接的输入或工作目录。
原运行保持原样；输入、已有产物和完整历史复制到新目录，来源及未完成调用记录在
`recovery.json` 中。Python 内存不恢复，模型需新建 Notebook 重新加载所需变量。
源目录和恢复目录都保留各自用量，汇总时按恢复链相加，不能将续跑误计为独立实验。

## 文本文件工具

`FileTools` 提供 `read_file`、`write_file`、`edit_file` 和 `apply_patch`，按会话显式登记。
这些方法共享当前会话的读取记录，模型不需要传递版本号或内容摘要。
路径以 `SessionState.workdir` 为根；前三个方法也接受该根目录内的绝对路径，
`apply_patch` 仅接受工作目录内的相对路径。

```python
from factor_demo.autoresearch_agent_v2.agent import Session
from factor_demo.autoresearch_agent_v2.tools import FileTools

session = Session(model, state)
session.register_stateful_tool(FileTools)
async with session:
    await session.call("write_file", {"path": "analysis.py", "content": "value = 1\n"})
    page = await session.call("read_file", {"path": "analysis.py"})
    result = await session.call(
        "edit_file",
        {"path": "analysis.py", "old_string": "value = 1", "new_string": "value = 2"},
    )
    assert result == "edited"
    # 文件工具也会提供给 session.run() 中的模型。
```

| 工具 | 参数 | 行为 |
|---|---|---|
| `read_file` | `path`, `offset=1`, `limit=200` | 返回指定范围的完整行，起始行从 1 计数 |
| `write_file` | `path`, `content` | 新建或完整覆盖，不追加、不合并、不自动增加末尾换行 |
| `edit_file` | `path`, `old_string`, `new_string`, `replace_all=False` | 精确字面替换；默认必须唯一匹配，空的新文本表示删除 |
| `apply_patch` | `input` | 用 Codex 格式补丁按序新增、更新、删除或移动多个文本文件 |

### 跨文件补丁

`apply_patch` 是 Python 原生工具，不启动 Node.js 或 Shell，也不依赖外部补丁命令。
注册 `FileTools` 或选择 `builtin:files` 时包含该工具；显式工具白名单需加入
`apply_patch`。Web 的研究模式允许该工具，对话模式保留原来的只读文件权限。

```python
result = await session.call("apply_patch", {"input": """*** Begin Patch
*** Add File: helpers/value.py
+VALUE = 1
*** Update File: analysis.py
@@
-value = 2
+from helpers.value import VALUE
+value = VALUE
*** End Patch"""})
```

补丁使用 `*** Begin Patch` 和 `*** End Patch` 包围。
`*** Add File: path` 后每行以 `+` 开头；空内容可新建空文件。
`*** Update File: path` 后用 `@@` 分隔修改区域；空格、`-`、`+` 前缀分别表示
上下文、删除和新增。`@@ context` 先定位上下文锚点；`*** End of File` 将区域锚定到文件尾。
更新标记后可紧接 `*** Move to: path`；`*** Delete File: path` 删除文件。
同一路径可在补丁中重复出现，后序文件操作基于前序结果；一个更新操作中的区域均在
该操作开始时的原文中定位。文件目标不能同时作为其他目标的父目录。

更新和移动前须读过源文件，删除前须完整读齐。补丁内部预检查读取不建立模型已读记录。
同一补丁内新建的文件可直接继续编辑或删除；局部已读文件经过更新后仍视为局部已读，
不能通过先更新再删除绕过完整读取要求。源文件与读取记录的内容摘要不符时，整批拒绝。
`Add File` 和移动目标必须不存在；迁移版不会覆盖已有目标。

匹配沿用 dsh/Codex 的顺序：原文、忽略尾空白、忽略首尾空白、标点归一化。
每轮取当前搜索范围内首个匹配，不要求唯一；重复内容应提供足够上下文。
这与 `edit_file` 的唯一字面匹配规则不同。
更新保留已有 UTF-8 BOM、统一 CRLF 和普通权限位；混合换行统一为 LF。
非空补丁更新结果按上游语义以换行结尾，移动目标继承源文件的普通权限位。

执行前检查整份补丁的语法、路径、文件类型、已读状态、内容匹配和版本。
预检查失败不创建目录、不修改文件。执行阶段每一步都复查目标快照，按序执行，
遇错停止；此前完成的操作不回滚。移动先发布目标再删除源，源删除失败时两者可同时存在。
执行失败还可能留下已创建的父目录。进程内锁不能消除外部进程的检查到发布竞争窗口。

成功返回 `status=applied`、`completed` 和 `affected_paths`，不回显正文或 diff。
失败返回 `error` 和 `hint`；预检查和执行错误分别带 `phase=preflight`、`phase=commit`。
执行错误还返回失败操作、已完成操作、受影响文件、`uncertain_paths` 和未执行操作数。
文件写入后结果无法确认时，相关路径进入 `uncertain_paths`，其读取记录失效；重试前重新读取。
单次补丁最多 1 MiB、128 个文件操作，单文件最多 16 MiB；快照和计划内容累计最多 64 MiB。

解析和文本匹配移植自 DeepSeek Harness `fs-patch` 提交
`39163dd30f703a79079e2d2446545cff3c453738`，其上游包含 OpenAI Codex Apache-2.0 代码。
版权、许可证和修改说明见文件工具目录的 `PATCH_LICENSE` 与根目录的
`THIRD_PARTY_NOTICES.md`。上游 85 个文本及错误定位样例和本地文件接入测试位于
`tests/test_patch_tools.py`；使用真实临时文件，不调用真实模型。

### 单文件读写

读取结果首行为 `lines 40-59/180`，随后每行使用“行号 + 制表符 + 原文”。
编辑参数不要包含行号前缀。结果包含未完成的分页时，末尾给出 `next_offset=60`。
每次最多输出 12 KiB UTF-8 字节，包含行号和分页信息；达到预算时在完整行边界停止，
不会触发通用的 20 KiB 工具结果落盘。单行无法放入预算时返回 `line_too_long`，
不把被截断的内容记为已读。空文件返回 `empty file`。

覆盖已有文件前，当前会话必须完整读取同一版本的内容。可以通过多次分页读齐，
缺失范围、失败读取和不同版本的分页不会建立完整读取记录。本会话新建或完整覆盖的
内容视为已知；完整已知版本的成功局部编辑也保留这一状态。只读过部分内容时，
局部编辑不会使剩余内容自动变成已读。

局部编辑前必须读过目标文件，或该文件由本会话成功写入。`old_string` 须精确匹配
当前文件，包括空格和缩进；默认连重叠位置也不能存在歧义。`replace_all=True`
替换全部不重叠匹配。原文、新文相同时返回 `unchanged`，不改写文件。
统一使用 CRLF 的文件接受 LF 编辑片段，并保留 CRLF；混合换行文件要求原样匹配。
读取隐藏 UTF-8 BOM，编辑及覆盖保留已有 BOM；写入完整内容时换行由 `content` 决定。
局部编辑保留文件其他字节、末尾换行状态和普通权限位。

外部变化使整文件覆盖返回 `file_changed`。局部编辑仍可在当前文件中进行唯一匹配，
成功时返回 `edited; changed_since_read=true`，并撤销原来的完整读取状态。
此后如果需要覆盖整个文件，必须重新读齐当前内容。

成功结果只有 `created`、`written`、`edited`、`edited N matches` 或 `unchanged`，
不回传全文或 diff。文件不存在、匹配不唯一、权限不足以及文件参数不合法等预期错误
以含 `error` 和 `hint` 的简短对象返回，不中断 AgentLoop。匹配不唯一时附带匹配数和
最多八个起始行号，不回显提交内容。其他工具默认的参数校验异常行为保持不变。

文件操作直接使用宿主授予的工作目录，不获得 Sandbox 的系统依赖或额外只读输入权限。
Web 每轮准备时，将当前登记数据集复制到新的 `workspace/dataset-*/` 目录，
再将副本的绝对路径登记为 `DATASET`；Router 和 Worker 使用同一副本。
原始版本仍保存在工作目录外，副本可以通过现有文件工具和执行工具访问。
复制拒绝符号链接，不覆盖已有输入副本或研究产物。
逐级目录描述符与 `O_NOFOLLOW` 拒绝父目录和文件符号链接，拒绝父目录跳转、目录、
FIFO、含 NUL 的内容及非法 UTF-8。单文件宿主处理上限为 16 MiB，与模型输出预算分开。
缺失父目录由 `write_file` 和补丁的新增、移动步骤创建；读取和单文件编辑不创建目录。

同进程、同根目录下的文件工具实例按文件共享锁，读取记录仍按 Session 隔离。
更新先完整写入同目录临时文件，检查目标版本，再原子发布；失败清理临时文件。
新建采用不覆盖已存在目标的发布方式。版本检查可以发现发布前已经发生的外部写入，
但与未使用这些锁的 Shell、其他进程或编辑器之间仍存在检查到替换的竞争窗口。
不要对同一文件同时启动外部写入。原子发布不等于跨进程事务，也不承诺断电后的目录持久性。
关闭 Session 会释放目录描述符并清除读取记录，已借出的工具也不能继续访问。
重新创建 Session 时需重新建立读取记录，不从对话文本推断文件版本。

实现参考 [Claude Code 文件工具](https://code.claude.com/docs/en/tools-reference)
的范围读取、精确替换和完整写入语义。12 KiB 输出预算、分页累计读取保护和简短成功
结果是本项目的设计选择。回归检查位于 `tests/test_file_tools.py`，文件操作使用真实
临时目录，模型响应使用模拟数据；这些检查不构成真实模型 token 节省率的测量。

## Shell 与 Python 沙箱

`autoresearch_agent_v2/sandboxing/` 提供独立的异步执行入口。由宿主把 Session 的
工作目录绑定到不可变的 `SandboxConfig`；模型只提交命令或源码，不选择授权目录和后端。
工作目录需要预先存在。每次调用启动一个独立的非交互 `/bin/sh`，文件保留在工作目录，
Shell 变量不跨调用保留。宿主仍可使用底层 `Sandbox.run_python()`；模型的 Python
验证统一使用 Notebook，不提供独立代码片段入口。通过有状态工具接入 Session：
启动时创建共享 Sandbox，跨对话轮次保留，关闭时撤销执行权限。

```python
from factor_demo.autoresearch_agent_v2.agent import Session
from factor_demo.autoresearch_agent_v2.sandboxing import Sandbox
from factor_demo.autoresearch_agent_v2.tools import SandboxTools

session = Session(model, state)
session.register_stateful_tool(SandboxTools)
async with session:
    sandbox = session.resource(Sandbox)
    assert session.resource(SandboxTools).sandbox is sandbox
    shell_result = await session.call(
        "sandbox__run_shell", {"command": "echo hello > result.txt; cat result.txt"}
    )
    # 同样的 Shell 工具也会提供给 session.run() 中的模型。
```

宿主需要自定义预算、只读输入或后端时，可在启动前改用
`SandboxTools.register(session, config=SandboxConfig(workdir=state.workdir, timeout_s=30))`，
其中 `SandboxConfig` 从 `sandboxing` 导入。此方式同时登记资源与有状态工具，
不再重复调用 `register_stateful_tool()`。配置目录必须与 Session 一致，
即使宿主提前提供了 Sandbox 实例，绑定时也会重新检查工作目录。

Shell 工具只提供 `command` 和可选的 `timeout_s`，不能提交 `workdir`、
后端或网络权限。不同 Session 各自持有 Sandbox；如果任务主动共享工作目录，
这些 Session 仍会共同读写其中的文件。关闭后的资源引用不能再次执行命令。
通过 Session 开始的操作遵循其关闭等待规则；宿主直接借用 Sandbox 开始的执行，
会在资源释放时取消并清理。此接入不替换旧版 PythonWorkspace。

`run_python()` 使用当前解释器的隔离模式 `-I`，通过 `shlex.join()` 引用源码和参数，
最终调用 `run_shell()`；不会把 Python 源码直接拼成可被 Shell 解释的文本。
两种入口共用目录权限、环境、超时、输出收集和清理逻辑。

| 配置 | 默认与含义 |
|---|---|
| `workdir` | 唯一可写的业务目录；运行依赖和只读输入不能放在其内部 |
| `read_paths` | 额外可读文件或目录，默认空；系统和当前 Python 环境自动只读授权 |
| `backend` | `auto`：macOS 选择 `macos`，Linux 选择 `bubblewrap` |
| `allow_network` | 默认 `False`；开启后允许联网，不提供目标地址白名单 |
| `timeout_s` | 默认 120 秒；调用时只能缩短，不能扩大 |
| `max_output_bytes` | stdout、stderr 各保留 256 KiB，超出后继续排空，避免堵塞 |

返回 `SandboxResult`：`backend`、`returncode`、`stdout`、`stderr`、`timed_out`、
`output_truncated`、`duration_s`；计算字段 `ok` 表示未超时且退出码为零，
也会序列化到模型收到的工具结果中。
取消执行会先清理再传播 `CancelledError`。后端缺失或平台不支持时抛出错误；
系统拒绝启动隔离环境、Landlock 安装失败等情况保留非零退出码及诊断，用户命令不会降级执行。

后端命令分别由 `backends/` 下的函数生成：

| 后端 | 实现方式与前提 |
|---|---|
| `macos` | 使用系统 `sandbox-exec` 策略，直接启动 Shell |
| `bubblewrap` | Linux 私有挂载、PID 和网络命名空间；只挂载授权输入、运行依赖及工作目录 |
| `landlock` | 在 bubblewrap 的私有命名空间中，由可信引导安装 NOOA Landlock 规则，再 exec Shell；要求 ABI ≥ 3 |

两个 Linux 后端都需要已安装且能创建命名空间的 bubblewrap，Landlock 模式不会退回无 PID
隔离的执行。该模块不导入旧版 `factor_demo/sandboxing.py`，也没有 `process` 无隔离后端。

执行环境不继承宿主凭据、`BASH_ENV`、`PYTHONPATH` 或动态链接器配置，并关闭继承的文件描述符。
授权目录仍可能包含敏感内容，因此宿主应只授权必要的输入；系统依赖采用目录级只读授权。
文件元信息不保证隐藏，`/dev/null` 等运行设备是业务目录之外的必要例外。

正常退出、超时和取消都会清理执行进程组；Linux 还依靠私有 PID 空间处理脱离进程组的后代。
macOS 的进程组清理不承诺回收主动 `setsid()` 脱离的后代，其文件和网络策略仍会继承。
目前提供墙钟和输出预算，尚未增加 CPU、内存、进程数量及工作目录磁盘总量配额，
也未复制旧执行系统的 seccomp 系统调用黑名单。

回归检查位于 `tests/test_v2_sandboxing.py`，覆盖 Shell、Python 包装、越界与符号链接、
只读输入、环境和描述符、联网开关、输出预算、超时、取消以及进程组清理。
Linux 命令构造检查与真实后端执行分开；在 macOS 上跳过 Landlock 集成检查。
`tests/test_session_sandboxing.py` 另检查会话绑定、跨轮复用、启动回滚、关闭撤权和真实工具执行。
大工具结果由宿主保存时，以 Session 工作目录为根使用目录描述符和 `O_NOFOLLOW` 逐级访问，
拒绝 Shell 创建的符号链接以及检查期间被替换的结果目录。

## Task、Agent 与 Session

一个 task 可以包含多个 session。每个 session 表示一个独立 agent 的对话，独立维护消息历史。
通用运行时创建子 agent 时直接继承父 agent 的 `workdir`，父子、兄弟及更深层子会话
共享同一个工作目录。文件、Shell 和 Notebook 中生成的文件可按路径直接互相读取和修改，
无需复制或发布。并行任务应约定不同输出文件；修改共享文件前先读取当前内容。
文件工具的读取记录仍按会话维护，完整覆盖时保留版本检查。
各会话仍独立持有 Sandbox、Notebook kernel、工具实例与模型历史；取消子会话不关闭父会话资源。
工作目录之外的文件、网络和凭据限制保持不变，运行时记录仍位于工作目录之外。
`SessionState` 只记录路径，本身不授予文件访问权限。

此规则适用于新建委派；旧版检查点恢复时保留已保存的目录，不自动迁移旧子会话产物。

| 字段 | 含义 |
|---|---|
| `session_id` | 独立对话的唯一标识 |
| `agent_id` | 本次对话所属 agent 的标识 |
| `parent_agent_id` | 父 agent 的标识；无父 agent 时为 `None` |
| `task_id` | 所属 task 的标识；多个 session 可以属于同一 task |
| `workdir` | 同一委派树共享的可写目录，类型为 `Path` |
| `messages` | 按顺序保存的完整消息字典列表，默认为独立的空列表 |
| `compaction` | 最近一次成功的摘要及其覆盖边界；默认为 `None`，随会话序列化恢复 |

## 完整消息历史

通过 `state.messages.append(message)` 追加完整消息，通过 `state.get_messages()` 获取历史副本。
返回值保留全部消息及其嵌套字段，不截断 `content`，不将消息简化为 `role` 和 `content`。
调用方修改返回值不会改变 session 中的历史。

消息可以包含 `reasoning_content`、thinking 内容块及其签名、工具调用、工具结果、多模态内容以及
供应商扩展字段。存储和读取时保留这些字段，以便后续模型调用继续使用原始消息结构。
消息结构由调用方提供；此数据类不校验具体模型协议，也不执行不同供应商之间的格式转换。
模型 SDK 对象应先由对应接入层转换为完整消息字典，再存入历史。

## 上下文压缩

`agent/history.py` 提供 `COMPACT_PROMPT`、`SUMMARY_PREFIX`、`compact_history()` 和
`get_context_messages()`。`state.get_messages()` 始终返回完整原始记录；AgentLoop 使用
`get_context_messages(state)` 构建模型请求，压缩成功后发送摘要和保留的近期轮次。
AgentLoop 每次任务请求前检查输入 token，达到配置的百分比时自动压缩；
宿主也可显式调用 `session.compact()`，不向 Agent 注册压缩工具。

`autoresearch_agent_v2/config.py` 中的 `HistoryConfig` 配置上下文预算和压缩策略：

| 字段 | 默认值 | 含义 |
|---|---|---|
| `context_window_tokens` | `172000` | 会话的上下文 token 预算，按实际模型和服务限制设置 |
| `compact_trigger_percent` | `80` | 输入占预算的该百分比时触发，范围为大于 0、至多 100 |
| `keep_recent_turns` | `2` | 压缩时保留的最近交互轮数；0 总结全部历史 |
| `compact_on_worker_reassignment` | `False` | 已有 Worker 接收新任务时，先将全部旧任务历史合并为摘要；首任务跳过 |

```python
from factor_demo.autoresearch_agent_v2.config import HistoryConfig

history_config = HistoryConfig(
    context_window_tokens=128_000,
    compact_trigger_percent=80,
    keep_recent_turns=2,
    compact_on_worker_reassignment=True,
)
async with Session(model, state, history_config=history_config) as session:
    await session.run("分析数据并保存结果")
    await session.run("比较两组差异")
    changed = await session.compact()  # 使用配置中的保留轮数，此例为 2。
    await session.run("继续检查未验证的假设")
    await session.compact(keep_recent_turns=0)  # 将当前全部历史合并为摘要。
    checkpoint = state.model_dump_json()
```

触发条件为 `input_tokens >= context_window_tokens × compact_trigger_percent / 100`。
默认预算下，输入达到 137600 token 时触发。
计数针对当前实际要发送的系统说明、数据集登记、有效历史、推理和工具调用信息；
原生模式额外计入工具 Schema，JSON 模式在完成消息转换后计数，避免重复计入工具定义。
使用现有 [LiteLLM token_counter](https://docs.litellm.ai/docs/completion/token_usage)，
未知模型使用库的默认 tokenizer，图片采用默认预算，不为计数读取远程图片。
这是输入估算，不累计整个会话的账单 token，也不包含本次尚未生成的输出。
实际服务端 tokenizer、图片处理和隐藏开销可能不同，配置的预算也不会自动探测模型容量。

压缩发生在完整工具批次返回之后、下一次模型请求之前，成功后重新构建并计数请求。
每次任务请求前最多压缩一次，达到阈值但没有可压缩前缀时继续保留近期轮次；
如果请求仍达到整个上下文预算，抛出明确错误，避免重复压缩或发送已知超过预算的任务请求。
可减少保留轮数或输入、工具说明，或按实际模型能力调整预算。
自动压缩失败向调用方传播，保留原检查点，不继续发出本次任务请求。
成功压缩时通过日志记录压缩前后的输入 token 估算。

压缩调用使用当前模型，沿用当前系统提示词、工具定义及待压缩的历史前缀，
将英文 `COMPACT_PROMPT` 追加为最后一条 user 消息；不再把历史重新包装为 JSON。
这样可在供应商支持时复用共同请求前缀；实际缓存命中与计费仍以服务端 usage 为准。
本次工具选择覆盖为 `auto`，避免普通任务的 `required` 强迫模型调用工具。
摘要响应不会进入工具执行循环；如果包含工具调用，则拒绝该摘要。
正常任务仍沿用原来的工具选择策略。
摘要覆盖较早的前缀，不拆开模型回复中的工具调用和整批结果。
一轮包含一条 assistant 回复及对应的全部工具结果；此前新收到的连续 user 消息并入该轮。
尚未收到回复的末尾 user 消息也按一轮保留，因此单个用户任务中的长工具循环可以逐轮压缩。
只有一轮且配置保留一轮时没有可压缩前缀；传 `0` 可总结全部历史。
工具结果缺失、孤立或批次被其他消息打断时拒绝压缩，先由宿主处理未完成调用。

英文压缩指令与摘要引导语采用 DeepSeek Harness `39163dd` 的原文。
指令要求保留八个章节：Primary Request and Intent、Key Technical Concepts、
Files and Code、Errors and Fixes、Pending Jobs、Current Work、Next Step 和 Critical Context。
摘要使用简短英文项目符号，保留准确路径、命令、错误、标识符、数值及用户纠正。
后续请求通过英文引导语与 `<compacted-summary>...</compacted-summary>` 包装摘要，
引导模型从摘要和后续消息继续工作，不重复回应检查点。
多次压缩合并已有摘要与新增历史，不叠加多个摘要；没有新增可覆盖内容时返回 `False`。
成功返回 `True`。模型错误、取消、空摘要、工具调用回复或未缩短上下文的结果均不替换旧检查点。
缩短检查比较序列化文本长度，不代表精确 token 计数或已测得的 token 节省率。

`Session.compact()` 与运行、工具操作共用会话锁；压缩函数还检查请求期间历史是否变化，
变化时拒绝发布旧快照的摘要。原始记录按只追加方式使用；宿主不应改写已被摘要覆盖的消息。
检查点随 `SessionState.model_dump_json()` 保存，恢复后继续使用同一摘要和消息边界。
`HistoryConfig` 由宿主传入 Session 或 AgentLoop；从状态创建新会话时应传回原压缩策略。
摘要是模型生成的记录，不能替代证据或证明执行资源已经恢复；Notebook 内存等仍需实际检查。

Web 入口通过 `configs/runtime.json` 的同名字段控制 Worker 换任务压缩，默认关闭：

```json
{
  "compact_on_worker_reassignment": true
}
```

该片段表示向现有配置加入或修改此字段。已有 Worker 的结果必须先收集，之后才能分配新任务。
开启后，`AssignTask` 仍立即返回；后台在同一 Worker 的新任务运行前调用
`session.compact(keep_recent_turns=0)`，不受普通 token 阈值或近期保留轮数限制。
全新 Worker 首次任务跳过压缩；持久化恢复的已用 Worker 也会在下一项任务前压缩。
旧检查点与尚未覆盖的历史合并为一份新摘要，新任务正文在压缩成功后才追加到有效历史。
原始消息和执行资源仍由原会话保留。没有新增可覆盖历史时不重复请求摘要。
压缩失败时，新任务产生可收集的 failed 结果；宿主取消则产生 cancelled 结果。
两者均不执行新任务正文，也不替换最后一次成功的摘要。
普通 token 阈值压缩仍独立生效。该开关不改变旧版 Agent 的摘要器。

客户端摘要会改变近期消息之前的历史前缀。已有 thinking 签名不能据此假定继续有效，
因此请求副本移除压缩时已存在的近期 assistant 消息中的 `reasoning_content`、`reasoning`、
`thinking_blocks` 及对应的 thinking 内容块。工具调用、结果与普通内容仍保留。
完整原始记录中的推理和签名不变，压缩之后新生成的推理也继续原样发送。

提示词来源为 [DeepSeek Harness summarizer.ts](https://github.com/deepseek-ai/deepseek-harness/blob/39163dd30f703a79079e2d2446545cff3c453738/packages/compaction/compaction-basic/src/summarizer.ts)。
版权和 MIT 许可保留在 `agent/COMPACTION_LICENSE`。
这是基于现有 LiteLLM completion 的客户端文本摘要，没有接入
[OpenAI 的加密 compaction 接口](https://developers.openai.com/api/docs/guides/compaction)
或 Claude 的签名 compaction 块。
回归检查使用模拟模型覆盖压缩、阈值边界、配置覆盖、单任务多次自动压缩、失败与取消、
完整记录保留、JSON 恢复、Worker 换任务压缩和 Web 配置传递，并使用本地 tokenizer 检查输入计数；
尚未验证真实模型的摘要保真度、签名兼容性、token 节省率或长任务表现。

## 模型调用

`agent/model.py` 中的 `Model` 提供异步接口
`completion(messages, *, tools=None, tool_choice=None) -> dict[str, Any]`，接收完整历史并返回一条完整的 assistant 消息。
`tool_choice=None` 沿用模型配置；压缩请求可单独覆盖为 `auto`，不改变后续任务的策略。
输入列表及其嵌套内容不会被本次调用修改。模型调用不自动追加 Session 历史，由调用方决定何时保存。

```python
from factor_demo.autoresearch_agent_v2.agent.model import Model
from factor_demo.autoresearch_agent_v2.config import ModelConfig

model = Model(ModelConfig(model="openai/model-name", api_key=api_key, base_url=base_url))
message = await model.completion(state.get_messages())
state.messages.append(message)
```

`ModelConfig` 显式接收模型标识和密钥；`model` 使用 `openai/` 或 `anthropic/` 前缀。
`base_url` 可指定兼容服务地址：OpenAI 通常使用带 `/v1` 的地址，Anthropic 使用根地址，
由 LiteLLM 补充 `/v1/messages`。`max_tokens` 默认为 `None`，本层不向请求传入输出上限；
接入库或服务端仍可能采用默认上限。单次请求的墙钟上限默认为 60 秒，包含连接与等待完整响应。
通过 `max_tokens`、`num_retries`、`timeout_s`、`thinking`、`reasoning_effort`
和 `tool_choice` 配置调用行为；
`thinking` 和 `reasoning_effort` 仅在显式设置时传入。
OpenAI 兼容服务的 `thinking` 扩展通过 `extra_body` 写入请求体。
`tool_choice` 默认为 `required`；DeepSeek 的 Chat Completions
thinking 模式不支持 `required`，参见 [官方接口说明](https://api-docs.deepseek.com/api/create-chat-completion/)。
密钥在配置展示及序列化时隐藏，实际请求时才取出。

输入和返回值使用 LiteLLM 的统一消息格式。返回整个 message 字典，保留
`reasoning_content`、`thinking_blocks`（含签名）、工具调用和 LiteLLM 提供的扩展字段。
Anthropic 原生消息由 LiteLLM 适配，因此统一格式与供应商原始 HTTP 响应不保证逐字段相同。
调用方应将返回消息整体保存，用于后续请求。参见
[LiteLLM thinking 消息说明](https://docs.litellm.ai/docs/reasoning_content)。

Web 核心 Agent 默认启用 `measure_api_ttft: true`，通过流式接收测量首个有效输出耗时，
然后合并为完整消息，Agent 循环仍按单条完整结果处理。
直接创建内核 `ModelConfig` 时，可设置 `measure_ttft=True` 启用。
统计写入每个会话检查点目录下的 `api_timing.jsonl`，Router 与 Worker 分别记录。
每次请求尝试具有独立 `attempt_id`；首个有效增量到达时记录 `first_token` 事件，
请求结束、失败或取消时记录 `end` 事件。
`ttft_s` 是从开始调用 SDK 到客户端收到首个有效输出的秒数，
包含客户端、网络及服务端等待，不是服务端内部生成耗时。
首个输出可以是思考文本、正文或工具名称/参数；空包、角色声明及纯用量包不计入。
`first_token_kind` 记录首输出类型，`duration_s` 记录单次尝试总耗时。
首输出前失败时 `ttft_s` 为 `null`，首输出后失败时保留已测耗时。
重试各自统计，不将退避等待加入下一次尝试的 `ttft_s`。
流式接口只能观测增量到达时间，不能识别服务器未向客户端发送的内部思考 token。
已有运行使用冻结源码，不会自动获得新增统计，也无法回溯首输出耗时。

适配器关闭 LiteLLM 与 SDK 的隐式重试，
并用独立计时器限制每次请求，避免底层超时仅约束读操作而让整次请求持续等待。
`num_retries` 默认为 `10`，表示首次请求失败后最多再请求十次，共最多十一次。
设置为 `0` 可关闭重试。只有超时、连接、HTTP 408/409/429 和 5xx 故障会重试；
鉴权、权限与参数错误直接返回，响应解析和用量回调错误也不触发重新请求。

指数退避从 1 秒开始，每次翻倍，单次等待最多 300 秒：
`1、2、4、8、16、32、64、128、256、300`。
`retry_initial_delay_s` 与 `retry_max_delay_s` 可修改等待参数。
任务取消或总时限到达会打断正在进行的请求及退避，不会因此发起新的重试。
错误最终未恢复时向调用方传播。

Web 的 Router 与所有 Worker 共用本轮总时限。`configs/runtime.json` 当前配置为：

```json
{
  "task_timeout": 7200,
  "router_time_budget": 2400,
  "router_token_budget": null,
  "request_timeout": 60,
  "request_max_retries": 10,
  "request_retry_initial_delay": 1,
  "request_retry_max_delay": 300
}
```

`task_timeout` 包含模型请求、重试、退避、工具执行及 Worker 等待。
Web 后台不再以固定 3600 秒覆盖此配置；显式传入的 `timeout` 仍可覆盖本轮总时限。
达到总时限后取消运行并保存检查点，状态记为 `timeout`。
`WaitWorkers` 的默认四小时等待上限仍受本轮总时限约束。

`router_time_budget` 是 Router 每轮任务的时间提醒预算，单位秒，默认 `2400`（40 分钟）。
从 Router 开始本轮执行时计时，模型请求、重试、工具执行及 Worker 等待均占用预算。
初始系统提示词包含带时区的开始时间、两项预算和时间预算的截止时间。
`router_token_budget` 默认是 `null`（无限），也可设为正整数。
token 用量累计本轮 Router 模型请求（含自动压缩）由服务端报告的输入和输出用量。
Worker 请求和未报告的用量不计入。
任一预算达到 `50%、60%、70%、80%、90%、95%` 时，
在下一次 Router 模型请求前追加 `system` 消息，显示两项预算的已用和剩余量。
同一次检查中两项达到同一个阈值时合并为一条；跨越多个阈值时一次补齐。
提醒在整批工具结果返回及自动压缩完成后追加。
模型请求或工具正在执行时不会被提醒中断，因而提醒可能晚于阈值时刻。
Worker 不单独生成预算消息；下一轮用户请求重新计时和计数，历史提醒仍保留。
耗时用单调时钟统计；超出预算时剩余时间显示为零，不强制终止任务。
宿主的 `task_timeout` 仍单独决定强制取消时间。
直接创建内核 `Session` 时，可通过 `TimeBudgetConfig(seconds=2400, tokens=None)` 设置
`time_budget_config` 参数；该配置位于 `autoresearch_agent_v2.config`。

无 Skill 公开案例的新入口为 `scripts/experiments/run_public_research_case.py`。
它要求新实验目录已准备好 `code/` 源码快照、`contexts/a|b/configs/runtime.json`
与 `results/<case>/<arm>/case/input/` 公开输入。两组配置只切换换任务压缩开关。
使用更新后的源码和运行配置，入口会按配置声明总时限；代码调用只计数，不再设置四十次上限。
新增入口拒绝覆盖已有状态的运行。上一轮冻结源码、脚本、提示词和结果均保留原样。

```bash
uv run --env-file .env python scripts/experiments/run_public_research_case.py \
  --suite runs/<新实验目录> --case 01-ai4i --arm a
```

## 多轮会话与单轮执行

`Session` 是多轮对话入口，持有 `SessionState` 和一个 `AgentLoop`。
`SessionState` 保存会话标识、工作目录、完整历史和数据集登记；`AgentLoop` 只执行当前一轮请求。
连续调用同一个 Session 的 `run(content)`，每轮都携带之前的历史。

内置消息工具接收 `content: str`：
- `finish(content)`：提交最终结果并结束当前轮次，正文作为 `run()` 的返回值。

```python
from factor_demo.autoresearch_agent_v2.agent import Session, SessionState

state = SessionState(
    session_id="session-1", agent_id="agent-1", task_id="task-1", workdir=workdir
)
session = Session(model, state)
result = await session.run("分析当前数据，并报告结果。")
follow_up = await session.run("继续解释刚才的结果。")
```

默认英文提示词定义在 `agent/prompts.py` 的 `SYSTEM_PROMPT` 中，保留用户确认的通用规则。
宿主根据 `parent_agent_id` 注入角色：无父会话的是入口 Router，有父会话的是 Worker。
Router 负责规划、委派、收集与汇报；研究所需的数据探索、计算及验证交给 Worker。
Worker 按任务中的交付清单返回文件及其用途。主子会话仍共享基础提示词与 Session 内核。

当前委派使用 `StartWorker`、`AssignTask`、`CheckWorkerStatus`、`WaitWorkers`
与 `CollectWorkerResult`。旧 sub-agent 工具组已删除。入口可创建多个并行 Worker，
所有 Worker 共享工作目录；会话上下文在同一根运行内保留，收集后可分配后续任务。
工具权限已按用户确认表实施，白名单集中在 `tools/role_policy.py`。
网页研究入口显式使用 Router 白名单，Worker 默认从宿主登记能力中选择批准子集。

| 工具类别 | Router | Worker |
|---|---|---|
| 五个 Worker 管理工具 | 开启 | 开启 |
| read_file、write_file、edit_file、apply_patch | 开启 | 开启 |
| Shell、Notebook、Python 工具类及 Statistics 方法 | 关闭 | 开启 |
| Skill、register_dataset、report、finish | 开启 | 开启 |
| report_progress、task_create、task_get、task_update、task_list | 关闭 | 关闭 |

权限同时约束模型 Schema、工具说明、直接调用和 Python 调用桥接。Router 不创建执行
沙箱或 Notebook 资源；Worker 从宿主配置取得自己的执行沙箱与批准的只读输入。
网页两类角色均不登记 task 工具。研究 JSON 使用文件工具保存，文件修改工具仍按原有
工作目录与读取校验运行。仅讨论模式采用 finish、report、read_file
三项能力，不创建或恢复 Worker。

恢复 Worker 时，保存的旧工具集合与当前宿主批准集合取交集，不能靠旧检查点恢复已关闭
的工具，也不会因宿主新增工具而自动扩权。底层通用 Session 仍接受宿主显式工具配置；
传感器对比脚本也按实际登记能力应用 Router 子集。

每次请求时以 `system` 消息放在历史最前面，
不重复写入会话历史。创建 Session 时可用 `Session(model, state, system_prompt="...")`
替换默认提示词。`before_run_hook` 仍只处理本次用户输入。

会话首次启动成功时，控制台通过 stderr 自动打印 session ID、完整 system prompt、已登记数据集的
名称/路径/描述、实际可调用的 tools 名称/说明，以及本次会话的 Python tools 类名。
完整提示词包含数据集说明、工具使用说明和 JSON 模式的调用协议，不截断内容。
没有数据集或未绑定 PythonTools 时，对应清单显示 `(none)`。Python tools 使用绑定时的
类目录快照；打印不会加载方法详情或提前创建业务工具实例。
`start()`、`async with session` 和首次 `run()` 均触发该输出；同一会话重复启动或继续对话
不会重复打印。启动失败不输出成功清单；后续动态登记的数据集会进入下一次模型请求。

提示词参考 Claude Code 的任务执行、沟通和结果报告方式，并按当前 loop 的工具协议改写。
参考资料包括 [Claude 官方的系统提示词定制说明](https://code.claude.com/docs/en/agent-sdk/modifying-system-prompts)、
[Claude Code 提示词提取仓库](https://github.com/Piebald-AI/claude-code-system-prompts) 和
[OpenAI 官方 Codex Prompting Guide](https://developers.openai.com/cookbook/examples/gpt-5/codex_prompting_guide)。
提取仓库是第三方从发行包中提取的片段集合，不是官方发布的完整系统提示词；
各片段可能对应不同版本或运行条件。本项目提示词是针对当前 Agent 的改写版本。
需要用户回答才能继续时，通过 `finish(content)` 说明阻塞并提出问题。

一轮执行中，每次将完整 assistant 消息和工具结果追加到会话历史，再请求模型。
同一响应中的工具按顺序执行；遇到 finish 结束本轮，后续工具不执行，
但为每个跳过的调用追加“未执行：finish 已结束本轮。”的工具结果，保持调用与结果配对。
finish 不销毁 Session，也不清空历史。下一次 `session.run(content)` 继续同一会话。
通过 `session.state` 读取会话状态，通过 `session.loop` 配置单轮循环。
已有完整历史也可作为 `SessionState` 传入新 Session 继续对话。
普通 assistant 文本不会自动结束循环。请求使用 `ModelConfig.tool_choice`；
若模型未返回工具调用，则保留原回复，追加一条 user 消息提醒模型调用工具，
并继续请求，直到模型调用 finish。提醒内容为：
“请调用工具继续；结束本轮请调用 finish。”
finish 参数解析或执行失败时返回工具错误，模型可继续修正。
在工具调用执行阶段，普通异常作为工具结果返回，
包含原始异常类型和消息，供模型决定后续操作；宿主权限与预算保持不变。
原生模式的 JSON 参数解析和参数校验错误也作为工具结果回传。
原生模式中，未注册工具名返回 `UnknownToolError`，工具内部的 `KeyError` 保留其原始类型。
Shell 执行超时或非零退出也返回执行结果，不结束 Agent 循环。
同一响应中的后续工具继续按顺序执行，不自动重试失败操作。
外部取消、进程退出等控制信号仍传播。模型请求、运行钩子或工具结果保存失败也向调用方传播。

### 统一工具调用格式

所有 provider 都通过 LiteLLM `acompletion` 接入。模型工具定义通过 `tools` 参数发送，
LiteLLM 将 provider 响应转换为统一的 assistant 消息，AgentLoop 只读取 `tool_calls`：

```json
{
  "role": "assistant",
  "content": "我先读取样本。",
  "tool_calls": [{
    "id": "call_123",
    "type": "function",
    "function": {
      "name": "read_file",
      "arguments": "{\"path\":\"sample.json\"}"
    }
  }]
}
```

`content` 可以是文字、JSON 字符串或 null，均不用于识别工具调用。
仅对 `function.arguments` 做 JSON 解码，按工具名称调用已注册工具。
调用 ID 沿用模型服务返回值，结果使用 `role="tool"` 和相同的 `tool_call_id` 回传。
推理、签名及 provider 扩展字段保留在消息历史中，由 LiteLLM 负责下一次请求的适配。
服务必须支持原生工具调用；正文中的伪工具 JSON 不会自动执行。

创建会话直接使用 `Session(model, state)`；旧的 `tool_call_format` 和
`max_parse_retries` 参数已移除，不再将工具 Schema 拼接进正文 JSON 协议提示词。
`ModelConfig` 当前支持 `openai/` 与 `anthropic/` 两种入口；其他 LiteLLM provider
并未因此自动开放。兼容 OpenAI 的服务使用 `openai/实际模型名` 并配置 `base_url`。

可用以下小型探针验证真实返回结构。它只发送人工构造的任务，读取本地函数生成的随机
标记，再要求模型通过 `finish(content)` 原样返回；不读取业务文件。默认使用已有 custom 配置。
每次最多四个模型请求、单次 90 秒、总计 180 秒、输出上限 2048 token，不自动重试。

```bash
uv run python scripts/experiments/probe_native_tools.py
uv run python scripts/experiments/probe_native_tools.py --provider custom --model glm-5.3-flash
```

`runs/native-tools-*/result.json` 保存 LiteLLM 转换后的完整 message、工具往返历史、
用量和通过状态，不保存凭据或服务地址。这是连通与协议检查，不代表复杂任务质量评估。

### 执行钩子

`AgentLoop` 提供两个异步钩子，可通过子类重写。钩子每轮执行一次：

- `before_run_hook(content) -> str`：在用户输入写入历史前调用，返回本次循环使用的内容，
  默认原样返回 `content`。
- `after_run_hook() -> None`：在本次运行退出时调用，默认不做处理。
  正常完成、异常或取消时都会执行；开始钩子抛出异常时也会执行。

```python
from factor_demo.autoresearch_agent_v2.agent.loop import AgentLoop


class MyAgentLoop(AgentLoop):
    async def before_run_hook(self, content: str) -> str:
        return f"请依据数据分析，并说明证据。\n\n{content}"

    async def after_run_hook(self) -> None:
        await resource.aclose()
```

`after_run_hook` 适合释放单轮资源。需要跨轮复用的资源应保留到整个会话结束，
不要在每轮结束时关闭。

新增工具注册到 `session.loop.tools`，支持同步和异步函数，注册后在后续各轮继续可用：

```python
def double(value: int) -> int:
    return value * 2

session.loop.tools.register(
    "double",
    double,
    description="将输入数值乘二。",
    parameters={
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
    },
)
```

注册的名称、说明和参数 Schema 会传给模型。工具结果以字符串写入历史，
其他返回值使用 JSON 序列化（保留 Unicode 字符）。写入前统一调用
`tools.result_guard.protect_tool_result(result, output_dir)`：UTF-8 编码后超过
20 × 1024 字节的结果完整保存到工作目录的 `tool_results/` 下，消息中改为 JSON，包含
`truncated`、`original_size_bytes`、`preview` 和绝对路径 `file_path`。
`preview` 共 500 字符，包含首尾内容和中间截断标记。恰好达到阈值时原样返回，不创建文件。
保存失败时异常向上传播，不将超限原文写入消息。文件随工作目录保留；文件读取工具另行实现。
`finish` 向调用方返回的正文仍保留原值，写入历史的工具结果遵循同一保护规则。
工具工厂按 Agent 实例保存，重复名称报错。

`tools/base.py` 中的 `ToolFactory` 提供注册和调用接口；
`create_tool_factory() -> ToolFactory` 返回独立工具工厂，默认注册 `finish`。
`AgentLoop` 创建时调用此函数；Session 在此基础上绑定当前会话的 register_dataset。

## 数据集登记

数据集通过名称、位置和简短描述告诉 Agent，由 Agent 使用已提供的执行工具和科学库按需读取。
宿主可以在启动前或会话运行期间登记：

```python
session = Session(model, state)
session.register_dataset(
    name="刻蚀观测",
    path="data/observations.parquet",
    description="每片一行的传感器汇总与量测结果。",
)
```

位置可以是文件、目录或 URI。相对路径以 `SessionState.workdir` 为基准。
名称、位置和描述均不能为空；长度上限分别为 200、4096 和 1000 字符，首尾空白会被去除。
同名登记更新原条目，其他条目保留。不同 Session 各自保存登记信息，即使它们使用相同工作目录。

Session 默认向 Agent 提供一个 `register_dataset(name, path, description)` 工具。
例如 Agent 生成新的特征表后，可登记该表的位置与用途。这个工具直接使用同一会话方法，
不需要额外启用工具组，也不提供查看、筛选、读取或切换数据集的专用工具。

每次模型请求时，系统提示词根据当前 `state.datasets` 重新生成一份登记信息。
Agent 在当前轮调用注册工具后，下一次模型请求立即看到新条目；后续对话沿用最新信息。
自定义系统提示词时也会追加登记信息。生成的提示词不重复写入消息历史，
登记字段作为 JSON 数据说明呈现，不作为新的执行指令。

登记信息随 `SessionState.model_dump_json()` 保存，通过 `SessionState.model_validate_json()` 恢复；
宿主仍负责保存和加载会话状态，不自动建立数据集数据库。旧状态没有 datasets 字段时默认为空。

登记只记录提示信息，不读取、复制、扫描或校验数据，不创建 DatasetSession 或 PythonWorkspace。
数据实际可读范围由现有执行环境决定；工作目录外的本地数据仍须由宿主通过
`SandboxConfig.read_paths` 授权，远程 URI 也不自动获得网络权限。
旧研究流程使用的数据版本、血缘和证据机制继续由原有模块负责，不接入这套轻量登记。

`tests/test_session_datasets.py` 检查宿主与 Agent 登记、同名更新、会话隔离、状态恢复、
参数校验，以及 native/JSON 两种协议中的动态提示词；模型响应使用模拟数据。

## Session 资源与有状态工具

`Session` 统一持有资源工厂、当前活实例、工具表和对话循环。
资源工厂在登记时不执行；`await session.start()` 或 `async with session` 才创建资源。
首次 `session.run()` 也会自动启动，以保留原有连续对话接口。
资源跨轮次保留，在 `await session.aclose()` 或退出会话上下文时统一释放。

```python
from factor_demo.autoresearch_agent_v2.agent import Session
from factor_demo.autoresearch_agent_v2.sandboxing import Sandbox
from factor_demo.autoresearch_agent_v2.tools import SandboxTools

session = Session(model, state)
session.register_stateful_tool(SandboxTools)

async with session:
    sandbox = session.resource(Sandbox)
    tools = session.resource(SandboxTools)
    assert tools.sandbox is sandbox
    # session.run(...) 继续使用现有 AgentLoop，资源不会在 finish 后关闭。
```

工具资源的权限由宿主配置，资源类型注册表不会自动创建新的文件或网络授权范围。

| 接口 | 作用 |
|---|---|
| `register_resource(ResourceType, factory)` | 登记接收 Session 的异步上下文工厂 |
| `register_stateful_tool(ToolsType)` | 登记有状态工具类，启动时统一调用其 bind 方法 |
| `await bind_resource(ResourceType, factory)` | 绑定期间取得共享资源，缺失时创建并交给会话清理 |
| `resource(ResourceType)` | 取得当前会话已经创建的同一实例，返回类型随类型键确定 |
| `schemas()` / `call(name, arguments)` | 从会话入口查看和调用工具 |
| `start()` / `aclose()` | 显式管理资源生命周期，运行中的 start 不重复创建 |

资源和有状态工具共用一个实例表。Session 先按登记顺序执行宿主资源工厂，
再按工具类的登记顺序调用 `bind(session)`，将返回的工具实例存入会话，
最后调用该实例的 `register_tools(factory)` 绑定模型可用的方法。
全部成功后，工具表才对外开放。

`StatefulTool` 协议声明三个成员：类级 `namespace`、类方法 `bind(session)`
和实例方法 `register_tools(factory)`。工具类只保存名称等声明；可变业务状态保存在
bind 创建的会话专属实例及其共享资源中，不保存在类变量中。
`bind` 返回异步上下文，其 yield 之前完成实例创建，finally 负责本工具独占资源的清理。
共享资源通过 `await session.bind_resource(ResourceType, default_factory)` 获取，
已有实例直接复用；缺失时只创建一次，并由 Session 在所有使用它的工具退出后清理。

多个工具类可以请求同一种底层资源，无需各自登记重复 factory。
若工具必须依赖另一个有状态工具，应先登记该工具，并通过 `session.resource(...)`
取得其已完成绑定的实例；缺失时明确报错，不隐式调用另一个工具的 bind。

`SandboxTools` 使用同一协议，可以直接调用
`session.register_stateful_tool(SandboxTools)` 使用默认配置；其
`SandboxTools.register(session, config=...)` 便利入口会先登记宿主配置的资源工厂，
再登记工具类，仍由 Session 统一调用 bind。

`ToolFactory.register_callable()` 使用 Pydantic 从带类型注解的函数签名生成参数
Schema，并在调用时严格校验参数，拒绝多余字段。同步或异步工具返回的 Pydantic
模型统一转换为 JSON 字典，供现有循环写入历史。原有显式提供 Schema 的
`register()` 接口继续可用；会话启动会保留此前手动登记的工具。

启动时使用内部工具表，等全部资源创建和工具绑定成功后才发布。
启动失败时按逆序清理已创建资源，并清空实例引用；工厂进入完成之前的部分资源
由工厂自己的上下文负责清理。启动后冻结资源登记表。
关闭开始后拒绝新操作，等待当前工具调用或对话轮次结束，再逆序释放资源。
正常退出、异常和通常的任务取消均触发上下文清理。
已关闭或启动失败的会话不再次启动，需要新建会话实例。

Session 顺序执行对话轮次和直接工具调用，不支持组间并行或工具嵌套回调
`session.call()` / `session.run()`。涉及真实工作区的选择与执行仍须使用其原有锁。
宿主通过 `session.resource(...)` 取得的引用只能在会话生命周期内使用，
不应绕过会话入口并发调用 `session.loop`。资源工厂如在清理时需要依赖，应保留创建时
取得的引用，不在会话关闭期间重新查询资源入口。

默认 Sandbox 由会话创建和关闭。借用宿主已有的 Sandbox 时，提前登记其工厂：

```python
@asynccontextmanager
async def borrow_sandbox(session):
    yield host_sandbox

session.register_resource(Sandbox, borrow_sandbox)
session.register_stateful_tool(SandboxTools)
```

宿主工厂在工具 bind 之前执行，因此覆盖工具提供的默认创建方式。
借用资源的工厂可只 yield，宿主继续负责其资源生命周期。
`bind_resource` 只在启动期间可用，运行期间通过 `resource` 获取已有实例。
活资源仅保存在 Session 中，不写入可序列化的 SessionState。

## 多 Notebook 工具

`NotebookTools` 在一个 Session 中按 `notebook_name` 管理多个真实 Jupyter Notebook。
名称允许中文，不包含路径或 `.ipynb` 扩展名。名称只在当前 Session 内查找，
Agent 不需要传递 Notebook ID。同名创建返回错误，不覆盖文件、不重置已有变量。

```python
from factor_demo.autoresearch_agent_v2.tools import NotebookTools

session = Session(model, state)
session.register_stateful_tool(NotebookTools)

async with session:
    await session.call("notebook__create_notebook", {"notebook_name": "数据探索"})
    cell = await session.call("notebook__write_new_cell", {
        "notebook_name": "数据探索",
        "source": "x = 40\nx + 2",
    })
    result = await session.call("notebook__exec_cell", {
        "notebook_name": "数据探索",
        "cell_id": cell["cell_id"],
        "revision": cell["revision"],
    })
    notebooks = await session.call("notebook__list_notebooks", {})
```

| 模型工具 | 用途 |
|---|---|
| `notebook__create_notebook` | 创建空文档，启动独立 kernel 并等待连接就绪 |
| `notebook__list_notebooks` | 返回当前会话的数量、名称、连接状态、文件路径和 cell 数量 |
| `notebook__write_new_cell` | 追加 cell，或使用 `after_cell_id` 插入到指定 cell 后 |
| `notebook__write_cells` | 按传入顺序一次插入 1 至 100 个 cell，整体保存，不执行 |
| `notebook__list_cells` | 按文档顺序列出稳定 cell ID、代码版本和执行状态 |
| `notebook__read_cell` | 读取源码、版本和上次执行输出 |
| `notebook__edit_cell` | 用 `expected_revision` 检查版本；以 `old_string`/`new_string` 局部替换，或用 `source` 整体替换 |
| `notebook__exec_cell` | 用明确的 `revision` 执行当前版本并保存输出 |
| `notebook__run_cells` | 按传入顺序执行 1 至 100 个明确版本的 cell，遇错停止 |

批量写入和执行示例：

```python
written = await session.call("notebook__write_cells", {
    "notebook_name": "数据探索",
    "cells": [
        {"source": "x = 40"},
        {"source": "x += 2"},
        {"source": "x"},
    ],
})
result = await session.call("notebook__run_cells", {
    "notebook_name": "数据探索",
    "cells": written["cells"],
    "timeout_s": 60,
})
```

`write_cells` 的 `after_cell_id` 对整批生效；省略时追加到文档末尾。
整批参数校验和插入定位完成后只保存一次，失败不部分写入。返回的 `cells`
每项只含 `cell_id`、`revision`，可直接传给 `run_cells`。

`run_cells` 执行前校验全部 cell 的存在性、类型和版本，任一不符则整批不执行。
执行顺序是参数列表顺序，不自动排序或补跑依赖。`timeout_s` 是整批共用的预算，
不能超过宿主上限；不是每个 cell 分别获得同样的时长。每个已执行 cell 单独保存，
批次返回 `status`、`executed_count`（已保存执行结果项数）、`connected` 和逐项 `results`。
Python 异常、超时、崩溃或执行期间的文件错误会停止后续项；未执行项使用
`status='skipped'` 和 `reason`，不修改其旧输出。普通异常保留 kernel，
正在执行的 cell 超时或崩溃会断开连接；两项之间预算耗尽时，空闲 kernel 保持连接。
文件保存失败时，Python 代码可能已有副作用，先检查状态再决定是否重试。
取消保留已完成结果，将当前 cell 标记为 `cancelled` 并继续传播，
尚未开始的 cell 保持原状态。批量执行不回滚已发生的 Python 变量和文件变更。

登记工具不会启动 kernel。`create_notebook` 成功返回时，Notebook 已连接；
不需要额外 open 或 connect。所有已创建的连接跨单轮结束和后续对话轮次保留，
Session 关闭时统一回收 kernel 及其同组子进程。不同 Notebook 拥有独立的 Python
内存状态，共享当前 Session 的文件工作目录。另一个 Session 可以使用相同名称，
其连接和文档路径仍然独立。

文档自动保存在工作目录的 `notebooks/<会话实例目录>/<名称>.ipynb`，包括代码、
nbformat 输出和执行版本。Session 关闭后文件保留，Python 内存释放。
当前工具只管理本 Session 创建的 Notebook，不扫描、打开或恢复以前的 `.ipynb`；
也未接入正在打开的 JupyterLab 页面协作。

创建和编辑都不执行代码。编辑后保留上次输出，并通过 `executed_revision` 与 `stale`
区分输出对应的源码版本。编辑函数定义不会自动替换 kernel 里已经存在的对象，
也不会重跑依赖 cell。`exec_cell` 只执行指定 cell 的当前版本，过期版本返回冲突。
`execution_count` 来自 kernel，表示执行顺序。

局部修改时，向 `notebook__edit_cell` 提供 `old_string` 和 `new_string`，无需重发整个 cell。
`old_string` 须非空且唯一匹配；需要替换全部匹配时，设置 `replace_all=true`。
`new_string` 为空表示删除。整体替换时仅提供 `source`，不与局部替换参数混用。
两种方式都必须提供 `expected_revision`；内容改变后版本递增，内容不变时版本不变。
版本冲突、无匹配或匹配歧义均不修改文档。修改后用返回的新版本执行 cell。

普通 Python 异常以 `status='error'` 返回，保留 kernel 及异常发生前的部分修改。
超时或崩溃返回 `timeout` 或 `crashed` 并断开该 Notebook；任务取消会清理 kernel、
记录 `cancelled` 并继续传播取消。其他 Notebook 的连接不受影响。
断开后不自动重启，以免把丢失的内存状态当作仍然存在；可创建新名称的 Notebook 继续工作。
当前未提供 `interrupt_kernel`。

kernel 使用现有 Sandbox 的文件、只读输入和网络权限，子进程不继承宿主凭据。
本地 Jupyter 通信使用 Unix socket；宿主仅额外授予该 kernel 专属临时通信目录，
不因连接需求开放网络。通信目录使用短路径以适配 Unix socket 的路径长度限制。
若同时登记 `SandboxTools`，两组工具复用同一个 Sandbox；如需只读输入或自定义预算，
先使用 `SandboxTools.register(session, config=...)` 登记宿主配置，再登记 NotebookTools。

单次执行受 `SandboxConfig.timeout_s` 和 `max_output_bytes` 约束。
`exec_cell(timeout_s=...)` 只能缩短宿主预算。超出输出预算时保留有效的有界输出，
通过 `output_truncated` 明确说明省略；超限消息不写入文档。
标准输出、表达式结果、HTML、图像和异常使用 nbformat 的输出结构。
`clear_output` 和同次执行内的 display 更新可处理；后台异步输出以及跨 cell 的
display 更新尚未同步到历史 cell。完整工具返回值仍经过现有工具结果保护。

回归测试 `tests/test_notebook_tools.py` 使用真实本地 kernel 检查多连接、变量保留与独立、
源码版本、文本和富媒体输出、默认权限、超时、取消、崩溃及关闭清理；
启动失败回滚另使用模拟失败，不涉及真实模型调用。

## Python 工具类与渐进加载

`python_tools/` 提供独立于现有 function calling 业务工具的类级接口。
启用后，初始系统提示词仅追加每个工具类的类级 docstring：说明整体能力，
并用一句话列出每个公开方法的用途。初始目录不包含业务方法的参数、默认值、
返回结构或完整方法文档，也不会提前构造业务工具实例或生成方法 Schema。

```python
from factor_demo.autoresearch_agent_v2.python_tools import PythonTools

session = Session(model, state)
session.register_stateful_tool(PythonTools)

async with session:
    await session.run("计算观测值的描述性统计。")
```

`PythonTools` 自动提供上节的九个 `notebook__*` 工具，无需再登记 `NotebookTools`。
此外只增加一个类发现入口，不提供 `python__execute` 或 `sandbox__run_python`：

| 入口 | 行为 |
|---|---|
| `python__describe_class(class_name)` | 按需创建指定类的会话实例，以 Python 类文档代码块返回全部公开方法的签名、类型注解和完整 docstring |

完整详情作为正常工具结果进入会话历史，后续请求可继续读取；初始类目录始终保持简短。
返回内容为带实际换行的 Python 文档，不使用 JSON 对象包装，也不追加原始 JSON Schema。
类中只展示接口与文档，不包含真实方法实现；内部继续根据类型注解校验输入和输出。
详情结果仍遵循现有大工具结果的落盘保护，超限时模型需要读取所返回的文件引用。
重复查看同一个类不重复创建实例。加载状态和工具实例在同一 Session 中跨轮保留，
不同 Session 即使共用工作目录，也不共享加载状态或实例。
从已保存消息创建新 Session 时，需要重新查看详情，旧历史不会自动授予调用权限。

内置 `Statistics` 的类级目录只说明 `summarize` 和 `pearson_correlation` 的用途。
查看详情后，模型才获得以下形式的 Python 文档。这里展示其中一个方法，
实际结果包含该类所有公开方法的完整 docstring：

```python
class Statistics:
    """对数值观测进行描述性统计和相关分析。

    方法：
        summarize：计算数量、均值、中位数、极值和标准差。
        pearson_correlation：计算配对观测的 Pearson 相关系数及检验结果。
    """

    def summarize(values: list[float]) -> dict[str, int | float]:
        """计算有限数值观测的描述性统计。

        Args:
            values: 至少一个有限数值；拒绝空列表、缺失值、NaN 和无穷值。
                不自动丢弃观测，不修改输入顺序。

        Returns:
            count、mean、median、minimum、maximum 和 std。
            std 使用总体标准差，即 ddof=0；单个观测的标准差为零。
            统计量沿用输入数值的单位。

        Raises:
            ValueError: 输入为空或包含非有限数值。
        """
        ...
```

业务方法优先使用 `str`、`int`、`float`、`bool`、`list[...]`、`dict[...]` 和 `None`
等原生类型，不通过自定义类型别名或 Pydantic 模型隐藏简单的入参、出参。
取值限制和返回字典的字段含义写在完整方法 docstring 中，并由方法校验业务约束。
通用调用入口仍根据类型注解严格校验参数和返回值，拒绝多余参数。

查看 `Statistics` 后，创建 Notebook，把代码写入 cell，再执行返回的明确版本：

```python
await session.call("python__describe_class", {"class_name": "Statistics"})
await session.call("notebook__create_notebook", {"notebook_name": "统计验证"})
cell = await session.call("notebook__write_new_cell", {
    "notebook_name": "统计验证",
    "source": 'result = Statistics.summarize([1.0, 2.0, 3.0])\nprint(result["mean"])',
})
result = await session.call("notebook__exec_cell", {
    "notebook_name": "统计验证",
    "cell_id": cell["cell_id"],
    "revision": cell["revision"],
})
```

代码中的类方法为同步调用形式，包括底层由宿主异步实现的方法。
入参与返回值经 JSON 传递；返回字典使用 `result["field"]` 访问，
不保留任意 Python 实例、DataFrame 或数组对象，数组可先用 `tolist()` 转为列表。
执行进程可使用沙箱内已有的 NumPy 等科学计算依赖。
通过 `print`、末尾表达式或富媒体展示返回结果。变量在同一 Notebook 中跨 cell 保留，
代码与输出保存到 `.ipynb`。需要独立片段或全新内存时，创建新的 Notebook；
不同 Notebook 的变量独立，同一 Session 的已加载工具类与宿主实例状态共用。
先创建 Notebook、后查看类详情也可以：下一次执行 cell 前会同步当前已加载类。
同步代理不增加文档 cell，也不占用用户代码的执行计数。

执行进程只接收已加载类的方法名和调用代理，不接收宿主实例、方法实现或模型凭据。
宿主在每次请求中再次检查类是否已查看、方法是否公开；直接伪造通信帧也不能绕过检查。
桥接使用 Jupyter stdin 通道和已有的本地 IPC，不为工具调用开启网络或扩大目录权限。
不请求用户交互输入；宿主只处理当前单元的授权工具请求。
`PythonTools` 与 `SandboxTools` 共用 Session 中的 `Sandbox` 资源，
需要自定义只读输入或超时等权限时，提前登记相同的 Sandbox 宿主工厂。
程序输出继续采用现有字节预算，单次工具请求和响应各限制为 1 MiB。

工具异常会在执行代码中表现为带原错误类型和消息的 `RuntimeError`，代码可捕获并处理。
未捕获的代码异常保存为 cell 的 nbformat error 输出，并返回 `status="error"`。
超时和取消会关闭受影响的 Notebook kernel，异步宿主方法会收到取消；
不自动重启或丢弃旧文档，需要继续独立验证时显式创建新的 Notebook。
同步宿主方法不能被强制终止，取消时等待它退出后再释放资源；
可能阻塞的自定义方法需要自行设置后端超时。

新增内置类放入 `python_tools/builtin_tools/`，在 `registry.py` 的显式模块清单中加载。
内置类与外部扩展均使用 `@PythonTool.register("ClassName")`：

```python
from collections.abc import Callable
from typing import Any

from factor_demo.autoresearch_agent_v2.python_tools import PythonTool


@PythonTool.register("Calculator")
class Calculator(PythonTool):
    """处理简单数值计算。

    方法：
        double：将数值乘二。
    """

    def methods(self) -> dict[str, Callable[..., Any]]:
        """声明公开方法。

        Returns:
            仅暴露给 agent 的方法映射。
        """
        return {"double": self.double}

    def double(self, value: float) -> float:
        """将输入数值乘二。

        Args:
            value: 要计算的数值。

        Returns:
            输入数值的两倍。
        """
        return value * 2
```

外部扩展模块需要由宿主在 Session 启动前显式导入，不自动扫描目录。
类方法必须通过 `methods()` 显式公开，其他公有方法及生命周期方法不会自动暴露。
默认首次查看详情时无参构造实例；有资源的类可重写异步上下文方法 `bind(session)`，
通过 `session.resource(...)` 借用启动时已准备好的共享资源，在 `finally` 中释放独占资源。
方法文档或类型声明不合格时，详情加载失败并回收新建实例，不开放部分接口。
Session 结束时统一关闭已加载类，并拒绝使用保留的旧入口引用。

`tests/test_python_tools.py` 同时检查模拟模型驱动的 Notebook 历史交互与真实隔离 Jupyter kernel 执行，
不将模拟模型回归视为真实模型的使用效果验证。


## 研究工作项工具

`TaskTools` 是保留给独立宿主显式装配的通用组件，网页 Router 和 Worker 均已关闭。
该组件提供 `task_create`、`task_get`、`task_update` 和 `task_list`。
主 Agent 可用它们维护研究步骤、记录进展和保存结果引用。先由宿主在 Session 启动前登记：

```python
from factor_demo.autoresearch_agent_v2.tools import TaskTools

# session 是已有会话；storage_root 是模型工作目录之外的宿主专用目录。
TaskTools.register(session, storage_root=run_dir / "host_state")
async with session:
    created = await session.call("task_create", {
        "subject": "检查特征排名的稳定性",
        "description": "完成重采样检查，报告 Top 10 特征的入选频率。",
        "active_form": "正在检查排名稳定性",
    })
    item_id = created["task"]["id"]
    await session.call("task_update", {"task_id": item_id, "status": "in_progress"})
    current = await session.call("task_list", {})
```

注册后，四个工具也会提供给 `session.run()` 中的模型。工具组采用显式登记；
现有调用脚本需要加入上面的 `TaskTools.register` 才会启用。存储目录由宿主选择，
不得位于 `SessionState.workdir` 内，也不要通过其他执行工具额外授权给模型。

| 工具 | 参数 | 返回与行为 |
|---|---|---|
| `task_create` | 必填 `subject`；可选 `description`、`active_form`、`metadata` | 返回 `task`，含 `id`、`subject`、`status`；初始状态为 `pending` |
| `task_get` | `task_id` | 返回完整 `task`；不存在或已删除时为 `null` |
| `task_update` | `task_id`，以及需要修改的字段 | 返回 `success`、`task_id`、`updated_fields`；状态改变时另含 `status_change` |
| `task_list` | 无 | 返回 `tasks`，按创建顺序包含全部未删除条目的 `id`、`subject`、`status`、`active_form` |

完整记录另含 `description`、`metadata`、`created_at` 和 `updated_at`，时间使用 UTC。
`subject` 与非空的 `active_form` 会去除首尾空白；标题不能是空白字符串。
默认描述为 `""`，执行文本为 `null`，元数据为 `{}`。普通状态为
`pending`、`in_progress`、`completed`，允许重新打开已完成条目，也允许多项同时进行。
`completed` 是 Agent 的完成声明，不验证文件、实验结果或研究结论。

更新只影响明确提交的字段。`description: ""` 清空描述，`active_form: null`
清空执行文本；其他字段不接受顶层 `null`。`metadata` 按顶层键合并：新键添加、
已有键整体替换、值为 `null` 时删除该键。嵌套对象不递归合并，数组整体替换，
空对象不改变元数据；创建时不接受元数据的顶层空值。非有限数字不允许保存。

删除使用 `task_update(task_id=..., status="deleted")`，不能同时修改其他字段。
删除保留历史，标识不复用；重复删除返回成功，其他修改返回 `task_deleted`。
无变化更新返回空 `updated_fields`，不改变修改时间。
参数错误返回 `invalid_arguments`，空补丁返回 `empty_update`，更新未知 ID 返回
`task_not_found`。参数校验在写入前完成；修改、审计和执行去重收据在同一事务提交。

计划保存在 `storage_root/task_plan.sqlite`，以当前 `SessionState.task_id` 隔离。
工具参数中的 `task_id` 是工作项 ID，例如 `T1`；与会话的研究任务 ID 含义不同。
同一存储目录和研究任务 ID 可以跨 Session 读取同一计划，不依赖聊天历史或压缩摘要。
恢复到新运行目录时，宿主需保留原存储位置或使用一致的数据库备份，并保留研究任务 ID；
仅复制 `session.json` 和工作目录不能恢复宿主存储。

原生模型调用会自动传递工具调用 ID。宿主重试同一次直接调用时，可以向
`session.call(..., call_id="稳定执行标识")` 传递相同标识。
去重范围是研究任务和会话：恢复同一次执行还需保留 `session_id`。
相同执行标识但参数不同返回 `execution_conflict`；未提供标识的直接调用、不同调用
以及模型重新生成的调用不会自动去重。同名标题允许多次创建。

回归检查见 `tests/test_task_tools.py`：数据库、并发修改和会话恢复使用真实临时存储，
模型响应使用模拟数据。这些检查不代表真实模型的计划质量评估。

## 从研究前端执行

启动 `factor-demo serve` 后，打开 `http://127.0.0.1:8765`。新建会话默认使用核心
Autoresearch Agent；选择数据集和模型，在“自主研究”中提交研究问题即可。
历史会话继续使用创建时的执行方式。

核心 Agent 的 task、Notebook、文件操作和最终回复复用现有事件通道，产物在研究面板查看。
每次请求结束后保存消息历史与任务计划；下一次请求恢复记录，但 Notebook 内存不跨后台进程
保留，需要重新加载变量。“停止研究”取消当前任务并保存检查点；“仅讨论”限制为读取已有
研究记录，不能执行分析或修改产物。该流程没有接入旧版 NOOA 的独立证据审查门禁。

## 通用工具注册与授权

内置工具位于 `autoresearch_agent_v2/tools/builtin_tools/`，提供文件、执行、会话委派、
技能和通用文档能力。业务工具位于独立的 `factor_demo/business_tools/`，由应用显式注册。
内核不扫描或自动导入业务包。

```python
from factor_demo.autoresearch_agent_v2.agent import Session
from factor_demo.autoresearch_agent_v2.tools.registry import ToolRegistry, register_builtin_tools
from factor_demo.business_tools import register_business_tools

registry = ToolRegistry()
register_builtin_tools(registry)
register_business_tools(registry)
session = Session(model, state, tool_registry=registry, allow_tools="all")
```

`allow_tools` 默认为 `"all"`，表示本次已注册并获准使用的工具全集，不会发现未登记的能力。
可传入明确工具名列表，或 provider 名，例如 `business:Statistics`。Python 业务方法使用
`Statistics.summarize` 形式的标识。未知工具在启动前报错。

授权同时作用于模型 Schema、工具说明、直接调用、Python 类文档和 Notebook 调用桥接。
部分授权的 Python 类只开放获准方法。权限名单不代替执行沙箱，也不禁止 Notebook 自行
使用已安装科学库；需要禁止代码执行时，应限制 Notebook 和 Shell 工具。

当前循环要求工具集包含 `finish`。
`allow_tools=[]` 或其他排除该工具的选择会在启动时失败，不会暗中重新加入工具。

工具定义和授权集合在会话启动时冻结。注册只保存定义与工厂，获准工具需要的资源才实例化。
`ToolRegistry` 按应用持有，Session 获得独立快照；后续注册不会修改已经构造的会话。

## Skill 加载

Router 与 Worker 使用同一基础 system prompt 和同一 Session 内核，宿主注入不同角色说明。
领域方法与研究字段来自 skill、用户请求和父任务；Worker 的通用交付外层由宿主校验。

框架没有默认启用的 Auto Research 业务 skill，也不预置固定科研流程。应用通过批准目录
显式提供技能，再用 Session 的 `skills`、运行时的 `worker_skills` 或技能加载工具选择。技能目录句柄由宿主持有，
不再使用该目录时可调用 `SkillRegistry.close()`；已冻结的会话正文不依赖部署文件继续存在。

本项目 Web 研究模式批准 `knowledge/v2/feature-research/SKILL.md`，仅 Router 可见和加载。
系统提示词中的技能目录仅包含 `name` 和 `description`，加载后也不拼接技能全文。
不默认预加载正文；当用户请求特征分析、根因分析、
故障定位等探索性分析研究时，由 Router 调用 `skill__load` 加载。普通任务不触发。
研究模式的 Router 开放 `skill__list`、`skill__load`、`skill__read_resource`；Worker
不开放这三个工具，其技能目录也不展示该技能，即使通过显式预加载或旧快照恢复也会被拒绝。
仅讨论模式不批准技能目录，继续使用只读工具子集。

宿主通过 `SkillRegistry(..., router_only=("feature-research",))` 声明角色限制，
实际角色由 Session 的父会话身份确定，不依赖技能描述中的自然语言约束。
恢复主会话时仅保留当前批准技能的快照，已停用的 `autoresearch` 和 `research-executor`
不会重新加载。Worker 不继承 Router 已加载的正文。

该技能由 Router 维护工作区内的 `research.json`，记录研究方向、依赖、结果、交付文件、
后续建议与集中独立验证。详细过程由 Worker 交付 Markdown 报告及可复现 Python 代码。
Web 运行时最多同时执行 3 个 Worker，包含数据探索、研究与验证任务。

```python
from pathlib import Path
from factor_demo.autoresearch_agent_v2.skills import SkillRegistry

skills = SkillRegistry([Path("/approved/skills")])
session = Session(
    model,
    state,
    tool_registry=registry,
    skill_registry=skills,
    skills=["analysis"],
)
```

批准根目录下的每个技能子目录包含 `SKILL.md`。YAML 头部使用 `name`、`description`、
可选 `version` 和 `requires_tools`。技能 ID 使用子目录名；正文与引用资源必须为 UTF-8 文本。
目录中拒绝符号链接，单个技能最多保留 128 个文本文件及 2 MiB 内容。

| 工具 | 参数 | 行为 |
|---|---|---|
| `skill__list` | 无 | 查看已批准目录与当前会话加载状态 |
| `skill__load` | `skill_id` | 幂等加载，通过工具结果返回全文，核对工具依赖并更新快照 |
| `skill__read_resource` | `skill_id`, `path` | 读取已加载快照中的正文或参考文件 |

首次请求前可按宿主显式配置加载 `skills` 中指定的技能，即使当前会话不获准调用技能发现或加载工具。
显式预加载的正文作为普通会话消息提供，不写入系统提示词；本项目 Web 入口的 `skills` 为空。
按需加载则由 `skill__load` 的工具结果返回正文，随正常工具历史保存和恢复，不在每次请求中重复拼接。
长参考资料通过 `skill__read_resource` 按需读取。

`SessionState.skill_snapshots` 保存版本、正文、引用文件与内容哈希。恢复时校验内容完整性、
当前目录授权及工具依赖，使用原快照，不静默替换为部署目录的新版本。技能加载不执行脚本，
也不会扩大 `allow_tools` 或文件、网络和凭据权限。

## 并行子会话与 finish 路由

`AgentRuntime` 管理会话树、并发槽位、持久化队列与父收件箱。没有独立的 research agent
或 verify agent 实现。每次委派创建通用 Session，使用独立历史、工具实例和执行目录。

```python
from factor_demo.autoresearch_agent_v2.runtime import AgentRuntime, RuntimeLimits

async with AgentRuntime(
    host_storage,
    limits=RuntimeLimits(max_concurrent=4),
) as runtime:
    runtime.attach(session)
    async with session:
        result = await runtime.run(session, "按用户目标完成工作")
```

运行时存储必须位于模型可写目录之外。默认 `SessionFactory` 复制父会话的工具定义、基础
prompt 与历史配置，不复制父消息或活跃 kernel。需要额外业务资源时，宿主使用
`SessionFactory(configure=...)` 显式装配，不能让模型指定数据库、凭据或工作目录。

| 工具 | 参数 | 行为 |
|---|---|---|
| `StartWorker` | `worker_type` | 创建空闲 Worker；批准类型目前只有 general，不开始任务 |
| `AssignTask` | `worker_id`, `task`, `deliver_list` | 分配给直属空闲 Worker；立即返回 assignment_id 与 D1、D2 清单编号 |
| `CheckWorkerStatus` | 无 | 查询直属 Worker 的 idle、busy、completed 状态，不收集结果 |
| `WaitWorkers` | 无 | 单独调用，等待任一结果；挂起时释放执行槽位 |
| `CollectWorkerResult` | `worker_id` | 收集完整结果并转为空闲，不等于验收通过 |

Worker 同时只有一个 Assignment。完成但未收集时不能分配新任务。相同上下文的修正、
补充可以复用已空闲 Worker；不同任务优先创建新的 Worker。每项 Assignment 有独立
标识与产物位置，历史结果不可覆盖。工具调用身份由宿主绑定，模型不能声明父会话、
工具权限、凭据或其他所有者。Worker 的执行能力由宿主独立配置，不要求是 Router 工具的子集。默认执行权限按
`tools/role_policy.py` 的确认表选择，宿主未登记的能力不会自动创建。

默认最多同时执行 4 项工作，委派深度上限为 2，根运行最多初始化 64 个 Worker。
空闲 Worker 与等待中的调用者不占执行槽位。WaitWorkers 默认最多等待 4 小时，结果就绪时立即返回，超时保留原任务状态；
没有待处理工作时立即返回 no_pending_work。Wait 不能和其他工具放在同一批次。
新用户引导可中断入口等待，不自动取消已派发任务。模型没有单独的取消工具，用户停止
与宿主运行控制负责递归取消。

Worker 通过 `finish(content)` 提交 JSON 对象文本，包含 outcome、summary、deliverables 与 limitations。
每个分配的交付 ID 恰好出现一次，每项包含 id、status、content 与 artifact_refs。
status 为 provided 或 unavailable，outcome 为 completed、failed、blocked 或 cancelled。
completed 要求全部交付项 provided；可靠的负面或不确定结论同样可以满足交付要求。
引用必须指向工作目录内真实存在的文件。交付记录上限为 12000 个 UTF-8 字节，长代码、
表格及日志放入产物。错误可修复，默认连续 3 次格式错误后由宿主生成 failed 结果。
结构完整不代表内容正确，调用者仍需按证据判断。

完成通知只包含宿主状态和结果标识。完整内容必须经 CollectWorkerResult 返回。
通知和 Wait 响应去重；即使通知已经送达，未收集结果仍会在后续 Wait 中返回。
收集回执与空闲状态一同保存；同一次调用重试返回固定 Assignment 结果，即使 Worker
已经执行了新任务也不会串结果。Start 和 Assign 同样保存幂等回执。

所有层级在调用 finish 前都必须等待并收集当前任务的子工作；否则返回 children_pending。
入口 Router 的 finish 正文面向用户，不要求套 Worker 的交付外层。失败、阻塞、取消也产生
可收集记录；completed 公共状态只表示执行已终止。

记录保存在宿主 runtime.sqlite3 的 worker_documents 表中，与旧版记录分开保存。
恢复已经完成的结果不会重新调用 Worker。未开始的直属 queued 任务可继续调度；
对已经运行、等待或无法确认副作用的中断执行，生成 failed 结果并保留检查点，
不静默重跑命令。旧 sub-agent 的历史记录保留，但不自动转换为可复用 Worker。
Notebook 内存只在本次根运行存活，不跨执行进程恢复。

## 研究 JSON 文件

主 agent 按 skill 或用户约定，在工作目录内维护真实的 `research.json` 文件，
或使用用户指定的可写路径。Router 使用 `write_file` 保存；有 shell 权限的 Agent
也可通过 `sandbox__run_shell` 写入。更新已有文件前先读取完整内容，文件发生变化时
重新读取并合并；写入后读取文件核对内容。保存失败时不得声称已落盘。

主会话是全局研究 JSON 的唯一写入者，这是 skill 约定。父子会话共享工作目录，
文件工具不提供按 agent 划分的研究文档所有权或文档版本号。

研究目标、证据、验收和后续方向的字段属于应用约定；运行队列、投递确认和执行状态
由通用运行时另行保存。旧运行的历史数据库保留，新研究不再创建独立文档数据库。

## 历史 verifier 的替代

旧 `feature_verifier/` 专用 Agent、`VERIFIER_PROMPT`、`feature_analyze_verifier` 和
`feature_verifier_status` 已退出核心架构。需要复核时，父 agent 下发新任务并选择适当 skill，
通过同一个通用委派工具运行，结果由 finish 返回。技能或用户可以约定复核格式，内核
不固定五折验证、性能改善状态或研究验收规则。

旧运行目录可保留为历史资料，不恢复已删除的专用 Agent 类。纯评估函数的历史保留与业务
工具迁移不改变科学算法，也不把可重跑或模型审查当作物理因果证明。

回归检查覆盖 `tests/test_tool_registry.py`、`tests/test_v2_skills.py`、
`tests/test_autoresearch_web.py`、`tests/test_agent_role_policy.py` 与通用运行时测试。
测试使用模拟模型及真实异步调度、文件和
SQLite 存储，不构成真实模型研究质量或 token 节省率的测量。
