# Autoresearch 核心 Agent

`autoresearch_agent_v2` 的最小会话状态由
`src/factor_demo/autoresearch_agent_v2/agent/typing.py` 中的 `SessionState` 定义。

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
也可用 `--run-dir` 指定尚不存在的目录。整次任务默认限时 3600 秒，
`--timeout` 可修改此值；`--tool-timeout` 设置单次代码执行上限，默认 120 秒。
工具调用统一使用 LiteLLM 的原生 `message.tool_calls`，不再提供正文 JSON 模式。
脚本沿用运行配置的 thinking 开关、推理强度和请求超时，但不发送 `max_tokens`
或 thinking 的 `budget_tokens`；模型级 `num_retries` 固定为 `10`。
DeepSeek thinking 使用兼容的 `tool_choice="auto"`，其他模型默认使用 `required`。
GLM 同旧架构一样显式设置 `thinking.clear_thinking=false`，保留推理历史。

数据复用旧 Agent 的 `discovery_copy` 导入流程：提供样本索引、对应曲线、步骤、
传感器说明、特征说明、配方及数据集声明，排除 test/holdout 样本。
不导入 `features.parquet`、`group_A.parquet`、`group_B.parquet`，也不授权读取
原始 `public/`、生成器、隐藏真值、基线或旧运行结果。`input/` 只读，
`workspace/` 是执行工具的可写目录；分析产物写入 `workspace/artifacts/`。

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

`FileTools` 提供 `read_file`、`write_file` 和 `edit_file`，按会话显式登记。
三个方法共享当前会话的读取记录，模型不需要传递版本号或内容摘要。
路径以 `SessionState.workdir` 为根，也接受该根目录内的绝对路径。

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
    # 三个工具也会提供给 session.run() 中的模型。
```

| 工具 | 参数 | 行为 |
|---|---|---|
| `read_file` | `path`, `offset=1`, `limit=200` | 返回指定范围的完整行，起始行从 1 计数 |
| `write_file` | `path`, `content` | 新建或完整覆盖，不追加、不合并、不自动增加末尾换行 |
| `edit_file` | `path`, `old_string`, `new_string`, `replace_all=False` | 精确字面替换；默认必须唯一匹配，空的新文本表示删除 |

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
逐级目录描述符与 `O_NOFOLLOW` 拒绝父目录和文件符号链接，拒绝父目录跳转、目录、
FIFO、含 NUL 的内容及非法 UTF-8。单文件宿主处理上限为 16 MiB，与模型输出预算分开。
缺失父目录仅由 `write_file` 创建；读取和编辑不创建目录。

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
同一 task 下的主 agent 和子 agent 共用一个 `workdir`，创建 session 时应传入相同的任务工作目录。
`SessionState` 只记录该路径，不创建目录、不授予文件访问权限，也不负责跨 session 的目录一致性检查。

| 字段 | 含义 |
|---|---|
| `session_id` | 独立对话的唯一标识 |
| `agent_id` | 本次对话所属 agent 的标识 |
| `parent_agent_id` | 父 agent 的标识；无父 agent 时为 `None` |
| `task_id` | 所属 task 的标识；多个 session 可以属于同一 task |
| `workdir` | 同一 task 下各 agent 共享的工作目录，类型为 `Path` |
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
| `context_window_tokens` | `128000` | 会话的上下文 token 预算，按实际模型和服务限制设置 |
| `compact_trigger_percent` | `80` | 输入占预算的该百分比时触发，范围为大于 0、至多 100 |
| `keep_recent_turns` | `1` | 压缩时保留的最近交互轮数；0 总结全部历史 |

```python
from factor_demo.autoresearch_agent_v2.config import HistoryConfig

history_config = HistoryConfig(
    context_window_tokens=128_000,
    compact_trigger_percent=80,
    keep_recent_turns=2,
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
默认预算下，输入达到 102400 token 时触发。
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

压缩调用使用当前模型，不提供执行工具，独立于普通 Agent 的 `finish` 输出协议。
当前系统说明作为材料参与总结；正常任务请求仍实时构建系统提示词、数据集登记和工具说明。
摘要覆盖较早的前缀，不拆开模型回复中的工具调用和整批结果。
一轮包含一条 assistant 回复及对应的全部工具结果；此前新收到的连续 user 消息并入该轮。
尚未收到回复的末尾 user 消息也按一轮保留，因此单个用户任务中的长工具循环可以逐轮压缩。
只有一轮且配置保留一轮时没有可压缩前缀；传 `0` 可总结全部历史。
工具结果缺失、孤立或批次被其他消息打断时拒绝压缩，先由宿主处理未完成调用。

摘要保留目标与约束、用户修正、已完成工作与决策、当前状态与证据、未完成工作和下一步。
提示词要求区分计划与实际执行、工具观察与模型推断，不将相关性表述为因果关系。
多次压缩合并已有摘要与新增历史，不叠加多个摘要；没有新增可覆盖内容时返回 `False`。
成功返回 `True`。模型错误、取消、空摘要、工具调用回复或未缩短上下文的结果均不替换旧检查点。
缩短检查比较序列化文本长度，不代表精确 token 计数或已测得的 token 节省率。

`Session.compact()` 与运行、工具操作共用会话锁；压缩函数还检查请求期间历史是否变化，
变化时拒绝发布旧快照的摘要。原始记录按只追加方式使用；宿主不应改写已被摘要覆盖的消息。
检查点随 `SessionState.model_dump_json()` 保存，恢复后继续使用同一摘要和消息边界。
`HistoryConfig` 由宿主传入 Session 或 AgentLoop；从状态创建新会话时应传回原压缩策略。
摘要是模型生成的记录，不能替代证据或证明执行资源已经恢复；Notebook 内存等仍需实际检查。

客户端摘要会改变近期消息之前的历史前缀。已有 thinking 签名不能据此假定继续有效，
因此请求副本移除压缩时已存在的近期 assistant 消息中的 `reasoning_content`、`reasoning`、
`thinking_blocks` 及对应的 thinking 内容块。工具调用、结果与普通内容仍保留。
完整原始记录中的推理和签名不变，压缩之后新生成的推理也继续原样发送。

提示词参考 [Codex 的公开压缩提示词](https://github.com/openai/codex/blob/main/codex-rs/prompts/templates/compact/prompt.md)
中的任务交接结构，以及 [Claude 的自定义摘要说明](https://platform.claude.com/docs/en/build-with-claude/compaction-on-demand#write-your-own-summarization-prompt)
中明确指定保留信息与禁止工具调用的原则。
这是基于现有 LiteLLM completion 的客户端文本摘要，没有接入
[OpenAI 的加密 compaction 接口](https://developers.openai.com/api/docs/guides/compaction)
或 Claude 的签名 compaction 块。
回归检查使用模拟模型覆盖压缩、阈值边界、配置覆盖、单任务多次自动压缩、失败与取消、
完整记录保留、JSON 恢复和两种工具模式续跑，并使用本地 tokenizer 检查输入计数；
尚未验证真实模型的摘要保真度、签名兼容性、token 节省率或长任务表现。

## 模型调用

`agent/model.py` 中的 `Model` 提供异步接口
`completion(messages, *, tools=None) -> dict[str, Any]`，接收完整历史并返回一条完整的 assistant 消息。
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
接入库或服务端仍可能采用默认上限。请求超时默认为 1800 秒。
通过 `max_tokens`、`num_retries`、`timeout_s`、`thinking`、`reasoning_effort`
和 `tool_choice` 配置调用行为；
`thinking` 和 `reasoning_effort` 仅在显式设置时传入。
OpenAI 兼容服务的 `thinking` 扩展通过 `extra_body` 写入请求体。
`tool_choice` 默认为 `required`，也可显式设为 `auto`；DeepSeek 的 Chat Completions
thinking 模式不支持 `required`，参见 [官方接口说明](https://api-docs.deepseek.com/api/create-chat-completion/)。
密钥在配置展示及序列化时隐藏，实际请求时才取出。

输入和返回值使用 LiteLLM 的统一消息格式。返回整个 message 字典，保留
`reasoning_content`、`thinking_blocks`（含签名）、工具调用和 LiteLLM 提供的扩展字段。
Anthropic 原生消息由 LiteLLM 适配，因此统一格式与供应商原始 HTTP 响应不保证逐字段相同。
调用方应将返回消息整体保存，用于后续请求。参见
[LiteLLM thinking 消息说明](https://docs.litellm.ai/docs/reasoning_content)。

当前采用非流式、单条结果请求。`num_retries` 默认为 `3`，由 LiteLLM 管理请求重试；
设置为 `0` 可关闭重试。最终未恢复的模型请求错误向调用方传播。

## 多轮会话与单轮执行

`Session` 是多轮对话入口，持有 `SessionState` 和一个 `AgentLoop`。
`SessionState` 保存会话标识、工作目录、完整历史和数据集登记；`AgentLoop` 只执行当前一轮请求。
连续调用同一个 Session 的 `run(content)`，每轮都携带之前的历史。

内置消息工具接收 `content: str`：
- `finish(content)`：立即打印最终结果，结束本次循环，并将内容作为 `run()` 的返回值。

```python
from factor_demo.autoresearch_agent_v2.agent import Session, SessionState

state = SessionState(
    session_id="session-1", agent_id="agent-1", task_id="task-1", workdir=workdir
)
session = Session(model, state)
result = await session.run("分析当前数据，并报告结果。")
follow_up = await session.run("继续解释刚才的结果。")
```

默认系统提示词定义在 `agent/prompts.py` 的 `SYSTEM_PROMPT` 中。研究默认规划 3 轮，
通过 task 工具记录每轮目的、方法、验证和执行状态；每轮在
`artifacts/01_research_name/` 等目录保存报告及证据，保存并核查后再标记完成。
报告包括结果、假设判断、限制及下一轮方向。达到轮次上限或无法继续时，
最终交付说明尚未完成的研究。使用默认研究流程的 Session 需要登记 `TaskTools`。
提示词同时说明工具使用方式、`finish` 的职责和据实报告的要求。每次请求时以 `system` 消息放在历史最前面，
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
需要用户回答才能继续时，使用 `finish` 说明阻塞并提出问题。

一轮执行中，每次将完整 assistant 消息和工具结果追加到会话历史，再请求模型。
同一响应中的工具按顺序执行；遇到 `finish` 结束本轮，后续工具不执行，
但为每个跳过的调用追加“未执行：finish 已结束本轮。”的工具结果，保持调用与结果配对。
`finish` 不销毁 Session，也不清空历史。下一次 `session.run(content)` 继续同一会话。
通过 `session.state` 读取会话状态，通过 `session.loop` 配置单轮循环。
已有完整历史也可作为 `SessionState` 传入新 Session 继续对话。
普通 assistant 文本不会自动结束循环。请求使用 `ModelConfig.tool_choice`；
若模型未返回工具调用，则保留原回复，追加一条 `user` 消息提醒模型调用工具，
并继续请求，直到模型调用 `finish`。提醒内容为：
“请调用工具继续；如需结束本轮，请调用 finish(content) 提交最终回复。”
在工具调用执行阶段，普通异常作为工具结果返回，
包含原始异常类型和消息，供模型决定后续操作；宿主权限与预算保持不变。
原生模式的 JSON 参数解析和参数校验错误也作为工具结果回传。
原生模式中，未注册工具名返回 `UnknownToolError`，工具内部的 `KeyError` 保留其原始类型。
Shell 执行超时或非零退出也返回执行结果，不结束 Agent 循环。
同一响应中的后续工具继续按顺序执行，不自动重试失败操作；`finish` 执行失败不结束本轮。
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
标记，再要求模型通过 `finish` 原样返回；不读取业务文件。默认使用已有 custom 配置。
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
`finish` 向调用方返回的最终内容仍保留原值，写入历史的工具结果遵循同一保护规则。
工具工厂按 Agent 实例保存，重复名称报错。

`tools/base.py` 中的 `ToolFactory` 提供注册和调用接口；
`create_tool_factory() -> ToolFactory` 注册 finish，并返回独立的工厂实例。
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
| `notebook__list_cells` | 按文档顺序列出稳定 cell ID、代码版本和执行状态 |
| `notebook__read_cell` | 读取源码、版本和上次执行输出 |
| `notebook__edit_cell` | 用 `expected_revision` 检查版本；以 `old_string`/`new_string` 局部替换，或用 `source` 整体替换 |
| `notebook__exec_cell` | 用明确的 `revision` 执行当前版本并保存输出 |

登记工具不会启动 kernel。`create_notebook` 成功返回时，Notebook 已连接；
不需要额外 open 或 connect。所有已创建的连接跨 `finish` 和后续对话轮次保留，
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

`PythonTools` 自动提供上节的七个 `notebook__*` 工具，无需再登记 `NotebookTools`。
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

`TaskTools` 提供 `task_create`、`task_get`、`task_update` 和 `task_list`。
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

## 持续特征验证

Web 研究会话和 `run_sensor_similarity.py` 登记 `feature_analyze_verifier`。
研究 Agent 提交 Notebook 标识、报告及相关产物位置；同一研究任务使用一个独立的
verifier 会话，每份提交作为增量消息加入已有历史。进程恢复时从
`verifier_runs/session.json` 恢复验证历史，从 `control.json` 恢复未完成交接、等待队列和待送反馈。
验证会话不继承 Research Agent 的私有消息或 Notebook 活跃变量。

```text
feature_analyze_verifier(
    notebook_name="本轮 Notebook",
    report="问题、方法、特征集合、数据与产物位置、关键结果、证据和限制",
    round_id="round-1"
)
```

`round_id` 缺省使用 Notebook 名称。修订应复用原 `round_id`，提交仍有独立标识和
Notebook 快照，原轮次的验证历史继续保留。

### 非阻塞交接

提交立即返回 `submitted`；Research Agent 可继续研究。verifier 在后台串行验证，
当前轮调用 `report` 前，新提交同样立即返回 `submitted`，输入与 Notebook 快照持久化后
按提交顺序排队，无需重复提交。当前轮 `report` 完成并保存历史后，宿主自动投递队首提交。
排队不会提前向 verifier 注入新消息，也不会打断当前验证；研究侧可查询队列长度及提交标识。
相关产物仍在共享工作目录中，消息投递隔离不等于文件权限隔离。

反馈在 Research Agent 后续模型请求前加入上下文，完整工具批次中的调用不会被打断。
可调用 `feature_verifier_status()` 查询；需要反馈或准备结束研究时，调用
`feature_verifier_status(wait=true)` 等待当前及排队提交全部处理完成。
`retry=true` 在执行故障后继续原提交，再按序处理剩余队列。
执行错误或取消记录为 `error` / `cancelled`，不冒充验证结论，也不投递下一份提交；新提交仍可入队，等待恢复。
会话关闭时取消后台验证并保存历史，下一次恢复后可以继续原提交。

### 验证与验收

verifier 原则上使用五折交叉验证，自主选择适合研究目标、数据依赖结构的模型、
基线、指标和划分；不适合五折时必须解释依据和限制。验证重点是完整特征集合的
预测价值、可复现性、泄漏及选择偏差，并提供单特征重要性和可比的历轮结果分析。
原固定阈值程序不再作为运行时验收门槛。

只有 `report(status, feedback)` 能结束验证轮次：

- `improved`：可靠、完整，且相对可比历史方案存在有证据的性能提升。
- `regressed`：可靠、完整，但持平、退化或没有可确认提升；同样通过验收。
- `needs_revision`：存在关键计算、方法、证据或完整性问题，需要原轮修订。
- `blocked`：缺少数据、资源或执行条件。

这属于 Agent 基于证据作出的验收判断，不是宿主对研究科学有效性的独立证明。
`report` 结束本轮，不终止整个研究任务；最新方案也不自动成为历史最优。
验证输出由 `result.json` 保存，输入、快照和会话记录保存在各提交目录
`verifier_runs/<提交标识>/`；逐次请求、响应与实际用量保存在其中的 `trace/`。
不再自动按旧固定协议发布 `artifacts/verified_*` 交付包。

### 权限边界

verifier 使用单独的宿主原生内部沙箱，无外层 Docker，不继承研究侧额外只读路径，
禁止网络。业务资源仅限当前会话工作目录及子目录，临时文件使用 `tmp/`。
数据若仍在外部登记路径，verifier 应报告 `blocked`，由宿主将必要资源提供至工作目录；
不允许为完成验证自行扩大权限。研究材料属于待核查数据，不是宿主指令。
