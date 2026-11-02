# Agent、工具与上下文设计

## 为什么以CodeActV2为主干

本场景需要模型根据中间证据决定下一步，临时编写统计或物理关系代码，并处理执行失败。仅让模型从预定义的几个筛选器中选择，无法覆盖原始曲线检查、估计器实现、批次置换和新因子构造。因此主研究使用原生CodeActV2与通用科学worker，算法自由度保留在研究层。

CodeActV2沿用NOOA代码执行、事件、配置校验与工具机制。外层现直接使用标准CodeActV2和SandboxedExecutor，能力探测只产生SandboxConfig；没有自行解析Markdown或替换REPL协议。方法可用 `_strategy=` 按本轮有效策略覆盖，避免通过修改共享策略对象影响同时运行的其他方法。

```{mermaid} diagrams/agent-loop.mmd
:caption: 主研究循环。确定性门禁和审查在候选交接处运行，正常科学探索仍由agent自由决定。
:alt: 当前任务经CodeActV2调用通用工具与独立分支，候选返回后核验、审查及有界修订，LoopGuard和预算约束探索。
```

## 研究方法和策略组合

| 方法 | 策略 | 作用与限制 |
|---|---|---|
| `research` | CodeActV2 | 自主研究，生成ResearchReply；运行前后有领域钩子 |
| `revise` | CodeActV2 | 针对当前报告与审查问题修订，返回完整候选，不重跑无关实验 |
| `chat` | CodeActV2只读工具 | Web讨论，可读产物、解释和reply，不执行研究或改文件 |
| `discuss` | Predict历史兼容 | CLI只讨论入口，不携带科学执行工具 |
| `review` | PredictStrategy | 审查给定报告和已复算证据，不独立运行实验 |
| `inspect_candidate` | 项目封装原生Reflexion | 只读局部实验；具体状态和原生判词单独返回 |

研究策略默认 `max_research_iterations=64`；局部反思基础策略为 6 次迭代，均还受整轮超时、模型次数/token预算、重复调用保护和单元超时约束。领域校验允许三次重试；这是执行/返回纠错次数，不是无限研究授权。最终审查最多两次定向修订，与这个重试计数不同。

## 公开工具

| 工具 | 输入与输出语义 |
|---|---|
| `run_code(code, timeout_s)` | 在当前研究轮的持久科学worker执行Python；返回状态、输出、epoch和宿主观察ID |
| `code_write`, `code_edit`, `code_read`, `code_list`, `code_run` | 版本化源码、CAS编辑、精确revision执行；编辑本身不改变内存 |
| `code_experiment`, `code_read_experiment`, `code_publish` | 显式依赖/参数和冻结输入的独立实验，检查后显式发布当前产物 |
| `read_observation`, `verify_repair`, `verify_goal` | 读父子执行证据、重放原失败路径、分别验具名目标；不能混同成功状态 |
| `list_executions`, `read_execution` | 读取科学执行来源、原始源码及有界输出，不重新运行计算 |
| `write_file(path, content)` | 只写workspace内文本/源码；PNG等二进制应在科学代码里保存 |
| `run_script(path, args, timeout_s)` | 运行已保存Python脚本，参数按字面传递，检查退出码 |
| `list_artifacts`, `read_artifact` | 列出真实产物或读取有界文本；不是凭空声明文件存在 |
| `reply(message_id, brief, image_paths, table_paths)` | 显式提供精确Markdown，宿主原样分段发送；不再调用第二个写作模型 |
| `spawn_research`, `list_research`, `wait_research` | 动态拆分问题、查询和等待独立方向 |
| `read_branch_result`, `guide_research` | 读取当前mission对应结果，或投递保留来源的父任务指导 |
| `cancel_research`, `resume_research` | 取消独立分支或用新mission显式恢复同一方向 |
| `inspect_candidate` | 只读审查候选；禁止计算、写研究文件、reply和任务变更 |

`run_code`默认120秒，上限300秒；执行超时会清理worker及后代，下一次需要重建内存状态。外层CodeAct单元超时330秒，局部Reflexion基础单元为60秒且最多六次CodeAct迭代。脚本失败须先检查返回值，不能仅凭工具调用正常返回就宣称实验成立。

## Prompt的职责分配

### 分析库按问题自由选择

Polars是表格读写、筛选和分组的首选，并不是整个研究过程只能使用的库。默认研究依赖包括下面这些能力；选择、组合或使用其他已安装库均由agent根据问题决定，不要求挨个调用。

| 库 | 可用于本场景的任务示例 |
|---|---|
| Polars / pandas | 宽表与目录整理、分组/连接、库间数据适配 |
| NumPy | 数组、线性代数、数值因子和时序窗口 |
| SciPy | 统计检验、信号处理、优化和科学数值计算 |
| scikit-learn | 预处理、PCA、建模、Pipeline和分组交叉验证 |
| aeon | 时序距离、变换、分类、聚类与相关时序方法 |
| statsmodels | 回归、统计推断、残差诊断与时序模型 |
| matplotlib | 保存真实研究图，交付PNG等产物 |

SciPy、aeon和statsmodels是显式项目依赖；实际版本由[锁文件参考](_generated/contracts.md)统一记录，而不要求用户手工拼兼容版本。aeon/Numba对NumPy和SciPy有上限约束，安装会选择整体兼容组合，不能把单独最新版本当作最佳环境。

aeon集合输入通常有case/channel/time三个维度；具体方法必须核对当前API和tags，不能把run和通道轴颠倒。statsmodels的显著性也不会自动解决lot伪重复、序列相关或自适应选择偏差。优先使用成熟实现，但不能把普通PLS冒充OPLS；确需自写算法时仍要做数值一致性检查。

worker为Numba JIT和joblib临时文件指定自身workspace下的缓存路径，继续保留文件/网络限制。核心依赖不等于安装了所有可选extras；遇到缺包先如实说明，由宿主补齐并锁定，不在隔离worker里联网安装。

Agent类文档表达行为契约：以证据产生和修订假设、区分因果层级、使用公开工具、返回结构化引用，并正确理解外层调度环境与科学workspace的区别。通用物理与统计规则由 `core.physics` 和 `core.statistics` 两个TextSkill承载，原生SkillRegistry通过agent.skills接入框架的原生目录表达式，按实例注册。`fixed_recipe`只表示数据约束，不选择工艺知识。专用领域包通过schema的`knowledge_profile`显式选择，加载部署目录中已安装的文本技能；未指定时保持通用规则。核心规则也进入稳定Context，并要求按需调用 `self.statistics.read_file('SKILL.md')` / `self.physics.read_file('SKILL.md')`。领域包启用时同样用 `self.domain_rules.read_file('SKILL.md')`。原生sandbox的 `doc(self)` 支持接口检查；`doc(self.<skill>)` 可能显示代理类型说明，不能用它代替技能正文读取。

通用物理技能强调真实单位、控制回路、守恒关系、延迟、测量原理和可观测性；工艺、配方约束、时段与step来自数据和用户目标。刻蚀术语与仿真候选机制只属于显式选择的`etch`领域包。统计技能强调lot独立性、监督选择泄漏、OPLS一致变换、有限置换p值下界、恢复率分母和闭环证据依赖。两者不是固定算法脚本，而是研究和审查必须遵守的解释边界。

## Context与自动摘要

```{mermaid} diagrams/context-budget.mmd
:caption: 原生上下文压缩与动态块。阈值是输入token数，不是输出长度，也不是累计会话token。
:alt: 272000总窗口预留50000输出，可用输入222000，177600触发原生摘要；完整历史仍在SQLite，稳定技能和动态任务分块注入。
```

稳定约束用前缀Context，任务、角色权限、当前DatasetRef/schema/选择栈单独注入；当前任务、最近十条专家反馈、最多十个近期假设及有界分支状态独立组织。较旧方向通过工具查询，而不是把所有报告无限重复塞进prompt。快照恢复后重新注入运行对象和技能，防止nosnapshot对象缺失或旧Context串用。

TokenBudgetSummarizer保留最近10条事件，目标摘要约8000字符。272000减50000给出222000输入预算，默认阈值是其80%，即177600。摘要调用同样经过预算包装；原始事件仍存NOOA SQLite，摘要不是删除审计证据。

历史运行曾实际触发一次原生摘要；这不能证明所有供应商的272k窗口都已经填满并通过压力测试。GLM当前配置使用thinking开启、effort=max；MiniMax Anthropic路由使用独立thinking budget，不虚构原生effort档位。

## 中文交接的现状

阶段性 `reply` 由agent主动提供Markdown；每次内部计算自动留记录，但不自动生成一篇长报告。候选报告保留假设、数学证据、替代解释与反证，最终正文由宿主从结构化结果和已核验事实确定性渲染。`StreamingReplyWriter` 仅保留兼容名称，不再发起第二次模型写作请求。纯可视化intent只做基础检查后直接交付；含假设或公式比较时不能走该快速出口。正文失败会交付已保存报告，并单独记录delivery_status，不重复研究。

尚未将“每次改算必须列传感器全名、step、旧值、新值、修改原因、结论变化”全部变成硬字段。传感器与特征名目前可包含在报告和证据中，但详细程度存在模型自由度。此缺口在[文档自审](quality.md)列为后续工作，不能用prompt要求代替已实现schema。

## Claude Code 对照实验

可以将已有研究会话的公开输入复制到新目录，用 Claude Code 调用同一套数据和科学执行工具：

```bash
uv run python -m factor_demo.claude_experiment \
  --baseline runs/original-session \
  --run runs/claude-comparison \
  --timeout 3600
```

需要预先安装支持 `--bare` 的 Claude Code。入口从原会话读取模型名称、任务和运行参数，从项目 `.env` 的 `FACTOR_API_BASE`、`FACTOR_API_KEY` 读取上游连接；endpoint 必须与原会话记录一致。它不会跟随 `.env` 当前的 `FACTOR_MODEL` 切换模型。历史会话未保存 key 指纹，因此只能保证复用当前指定凭据来源，不能追溯证明旧运行的密钥未变。

本地 LiteLLM 转换层将 Claude Code 的 Anthropic Messages 请求送往原 OpenAI 兼容服务，固定模型、输出上限、thinking 和 reasoning effort。上游凭据只在宿主转换层使用，CLI 获得临时本地凭据，MCP 工具进程启动后清除凭据环境变量。Claude Code 只获得显式 MCP 业务工具；未开放其内置文件、终端、网络和子 agent 工具。科学计算仍使用原有 `PythonWorkspace` 隔离执行。

每次只运行一个主 agent、一个并发模型请求，不启用额外模型审查；最多 64 轮、256 次模型请求、累计 400 万 token，token 预算在请求边界检查，单次请求可跨越累计上限。原会话的全部公开文件通过 SHA-256 核对，旧分析和隐藏真值不复制。先复现宿主数据预检，再用同一 Claude Code 配置执行独立最小模型探针，通过后才开始正式任务。

新目录保存 `manifest.json`、`message.md`、`system-prompt.txt`、科学执行记录，以及 `probe/`、`formal/` 下的 `tool-calls.jsonl`、CLI 流、网关请求、上游 usage 和耗时。探针单独统计。最终回答保存在 `final-response.md`，分析产物位于 `workspace/artifacts/`。CLI 正常结束只表示模型交接完成，结果正确性还需核对实际表格、代码和证据。

可在运行中或完成后重建对比统计：

```bash
uv run python -m factor_demo.claude_experiment.report \
  --baseline runs/original-session \
  --run runs/claude-comparison
```

统计保存在新目录的 `comparison.json`。NOOA 从原始 trace 读取完整响应 token，Claude Code 从上游 usage 读取；两边均排除宿主预检，Claude 另排除工具探针。输出 token 已包含思考 token，不能将两者再次相加。

这是框架层对照：两套执行方式的系统提示、上下文管理、工具协议和最终交接不同，不能据此声称是严格的单变量模型能力评测。比较时分别报告任务完成、生成 token、重复代码、执行失败和产物一致性；网络等待时间和 API 故障另列。


## 专家公式研究工具

原生CodeActV2可调用create_formula_study、begin_formula_comparison、register_formula、evaluate_formula_set、read_formula_study、retire_formula_set及freeze_formula_selection。只读讨论与Reflexion禁止变更；Context跨轮注入研究ID、版本和选择状态。仅本轮创建、修改、执行或显式继续比较时，postcondition要求有效选优或明确未完成；只读、普通画图和新轮任务不受历史公式研究阻塞，Predict获得宿主证据，正文最终提交附带确定性公式报告。任意Python算子和四类固定比较模型的边界见[公式研究](formula-research.md)。

## 动态范围与有界状态

CodeAct用inspect_dataset探索schema，用任意Python生成划分/合并/变换脚本，derive_dataset登记后select_dataset或return_dataset控制范围。领域规则不会把普通表格、文本、图片伪装成刻蚀数据。分支Context声明can_spawn，宿主限制子分支递归派生。公式历史仅摘要最近五项与本轮活跃研究，每项最多八个近期版本/试验；原始记录可用工具查询。

共享token账本按请求内容和配置输出上限预留，并保留未知请求的保守回退；272k上下文、50k输出、thinking/effort及原生摘要阈值保持既有配置。修订次数耗尽显示revision_limit，预算耗尽显示budget_exhausted，两者不能混写。

工具说明明确了DATASET/DATASET_ID也注入run_script，执行文本返回字段为output；EvidenceRef.run_ids只能填写数据集中的真实样本身份，数据集级计数可留空。reply.message_id是自行选择的稳定发布键。未知契约要求先读文档或做小步调用，减少在执行前猜测接口与长篇预演。

## 执行反馈与源代码纠偏

工具表是按用途的摘要，完整参数和当前源码说明见[自动生成工具契约](_generated/tools.md)。`execution_state` 动态块同时包含主内核状态、独立实验的 current/stale 摘要和宿主未解决失败。错误上下文以 observation ID 读取完整记录，不依靠模型回忆几句报错。

源码编辑、执行、原路径复验和目标断言是不同状态；`verify_repair` 不是把“新函数定义成功”直接记为修复，`verify_goal` 也不替代报告的物理/统计门禁。局部反思接收同一证据快照但只能读。运行边界和完整流程见[持续REPL与纠错闭环](repl-runtime.md)。
