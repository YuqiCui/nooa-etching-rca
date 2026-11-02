# 持续 REPL、独立实验与纠错闭环

本章描述当前已经合并的执行协议，不是待实施计划。持续执行、源码编辑、实验隔离和失败复验各自承担不同职责：保留活对象不等于保存源码，保存源码不等于执行成功，执行成功也不等于研究目标成立。

## 当前实现与验证范围

| 能力 | 实现入口 | 已验证的边界 |
|---|---|---|
| 持续解释器 | 科学 `PersistentKernel`、NOOA `SandboxedExecutor` | 同一生命周期内共享 namespace 和事件循环，保留函数全局引用、生成器和锁；不靠 pickle 重建 |
| 精确版本源码 | `CodeBuffers` | 比较并交换式写入、唯一文本替换、明确 revision 执行；编辑不修改正在使用的函数 |
| 可丢弃实验 | `CodeExperiments` | 独立 worker、冻结输入副本和 JSON 参数；失败不改变主 REPL 和正式产物 |
| 宿主执行证据 | `ExecutionObservations` | 外层/内层父子关系、源码、异常、输出与状态；外层随后失败不丢已发生的内层记录 |
| 原路径修复复验 | `verify_repair` | 相同输入范围和字节、保留原失败检查；无关成功不能解决旧失败 |
| 独立目标检查 | `verify_goal` | 重算一个明确的目标断言；不自动发布、不代替全部科学正确性审查 |

执行协议由 [PR #10](https://github.com/pty819/nooa-etching-rca/pull/10)、[PR #11](https://github.com/pty819/nooa-etching-rca/pull/11)、[PR #12](https://github.com/pty819/nooa-etching-rca/pull/12) 引入；[PR #13](https://github.com/pty819/nooa-etching-rca/pull/13) 将依赖单独升级为 NOOA `0.0.11.dev473`。当前平台和真实模型验收见[验证策略](validation.md)，不能从机制存在推断任意模型都能自行完成研究。

```{mermaid} diagrams/execution-correction.mmd
:caption: 编辑、隔离实验、原路径复验、目标检查和发布是不同动作。
:alt: 源码编辑产生新版本，隔离实验给出宿主执行证据；失败进入下一轮上下文，修改后重放原检查，再单独验目标并显式发布产物。
```

## 原生执行器与能力配置

当前外层直接使用官方CodeActV2和SandboxedExecutor，旧SecureCodeActV2/ExternalCell、子进程代理及JSON协议已移除。能力探测只选择官方SandboxConfig，不自行重建namespace、helper或返回协议。环境变量与REST配置见[运维说明](operations.md)。本章旧PR验收属于历史版本；原生迁移的验证范围单独记录，不冒充旧边界仍全部成立。

## 两层长驻对象的寿命

外层 `python_cell` 在同一次 NOOA 生成调用内复用官方fork worker；科学 `run_code` / `code_run` 在当前研究轮次的 `PythonWorkspace` 中复用另一独立解释器。二者通过工具协议通信，不直接共享 Python 对象。官方外层保持其支持的活对象语义；科学函数、DataFrame、生成器和锁可以留在科学namespace；科学函数重新定义后读取的是当前 namespace 的全局绑定。

`run_script` 使用独立临时内核，不继承交互变量；`code_experiment` 则从显式源码、参数和冻结数据重建实验输入。它们结束后关闭自身 worker，不清空主科学 REPL。

新用户请求、显式恢复或数据范围切换会建立新的科学内核。NOOA 对话和源码历史可以恢复，内存数组与函数不会自动恢复。当前没有跨 HTTP 请求的长寿命 SessionManager，也不会自动重放旧代码或外部工具副作用。`kernel_id` 和 `epoch` 应一起读取；换一个内核对象时不能把 epoch 当作整个会话的全局递增序号。

## 失败后内存如何处理

| 事件 | 主科学 REPL 的处理 | 不能做出的推断 |
|---|---|---|
| 编译/语法失败 | 未执行用户源码，保留现有 namespace | 保存了新源码不代表新定义已生效 |
| 普通运行异常，包括执行中主动抛出 SyntaxError | 关闭受影响内核，下次执行重建 | 报错前的对象修改没有自动事务回滚 |
| 超时、取消、崩溃、协议或清理失败 | 使受影响内核失效并保留诊断 | 退出码为零但没有匹配完成帧不能算成功 |
| 临时脚本或隔离实验失败 | 关闭临时 worker，主内核不受该失败影响 | 已写入临时目录的文件不等于正式证据 |
| 外层 cell 运行异常 | 按NOOA原生规则处理，普通异常可保留namespace；超时/崩溃按上游恢复配置重建 | 不应把科学内核“运行失败即重置”的规则套到原生外层；工具副作用不会撤销 |

运行失败重置不能撤销文件写入、工具调用或共享外部系统的副作用。高风险试错应放入独立实验，不能靠删除几个变量就声称现场干净。Linux 科学 cell 成功返回前还会清理其启动的子进程，保留内核引导链；活 Python 对象可以持续，后台子进程不能跨 cell 存活。CPU 上限按内核寿命累计，墙钟截止时间按 cell 检查。

## 独立实验与产物有效性

`code_experiment(name, revision, checks, dependencies, parameters)` 要求候选和依赖使用明确正整数 revision。依赖按传入顺序执行，然后执行候选，最后执行非空检查源码；总墙钟预算覆盖整个链路，默认 120 秒、上限 300 秒。`PARAMETERS` 只接收 JSON 值，`DATASET` 指向宿主核对哈希后复制的当前数据版本。主 namespace、其他工作区文件和凭据不继承。

结果的 `status` 与 `validity` 是两条轴。`checked` 只表示提交的检查源码正常完成；`failed`、`timeout`、`cancelled` 保留诊断。`current/stale` 则比较全部依赖/候选 revision、当前 DatasetRef 和输入文件字节。即使新旧源码文本相同，只要 revision 变了，旧结果也会失效；旧 revision 仍可显式复跑，但不能当作当前版本发布。

`code_read_experiment` 返回宿主保存的精确源码、参数、检查、执行列表和有效性。通过检查后，`code_publish` 才能把当前产物复制到唯一的 `artifacts/experiment-<id>/` 目录。它拒绝失败、过期、暂存文件变化和符号链接；不会覆盖其他实验目录，也不会激活主 REPL 中的函数。发布后的文件若被修改或源码/数据变更，宿主证据校验仍会拒绝其引用。

这里实现的是保守的整组源码与输入失效规则，不是任意 Python 对象的自动依赖图。普通 `run_code` 写文件不会自动获得同样的候选发布事务；选择哪种执行路径应明确。

## 失败、修复与目标分开验收

实验失败后，下一轮应依据 observation ID 阅读原源码和异常，编辑新 revision，然后调用 `verify_repair`。对于实验，它要求候选/依赖名字和顺序相同，至少一个 revision 改变，保留原 JSON 参数并重用宿主保存的原 checks。对于科学 cell，它在显式候选定义后完整重放原失败源码。只有当前版本在相同 DatasetRef 与输入字节上通过，才记录修复关系并解决该失败及其失败子记录。

自包含的旧错误源码（例如直接写死 `1 / 0`）不会被工具偷偷修改成一个容易通过的检查；可能需要先记录一个调用候选函数的复现实验。脚本复验也没有隐式工作区文件重建能力。工具验证的是原路径重跑结果，不推断任意程序语义等价或修复的因果必然性。

`verify_goal(goal, experiment_id, checks)` 从一个当前已检查实验重算明确的目标断言，返回 `supplied_criterion_passed`。它不清除其他失败、不发布文件，也不保证断言完整表达用户全部目标。失败检查会继续可见；后续一次成功不能自动抹去另一项反证。完整例子见[版本化代码](code-buffers.md)。

## 证据进入下一轮与只读反思

`.execution/observations.sqlite` 保存宿主观察和修复关系。动态 Context 提供最近 3 项观察、最多 6 项未解决失败及总数；源码和输出是有界首尾预览，可用 `read_observation` 继续读取。外层异常单独归档为编排诊断，不与未解决科学/实验/检查失败使用同一个计数。普通成功不删除旧失败。

`inspect_candidate` 把同一份 `execution_context()` 作为 `execution_evidence` 参数传给只读反思。反思可读取已有观察，却不能执行实验、编辑、发布、发送回复或派生研究。它仍是同模型局部审查，不是独立重算。完成协议来自不可信 worker；宿主记录来源和实际流程并不能抵御任意同进程恶意 Python，也不能认证数学结论。
