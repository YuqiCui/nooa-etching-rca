# 验证策略与能力证据

## 四层证据分别回答不同问题

```{mermaid} diagrams/validation.mmd
:caption: 工程可靠性、真实模型联通与科学结论正确性分别验收，最后才形成文档可声明的能力。
:alt: 源码由物理与契约测试、真实NOOA集成及Vue测试检查，限定真实模型验收和历史科学真值对照共同形成审查和发布依据。
```

单元测试回答数值和边界规则是否按实现运行；集成测试回答NOOA、HTTP与独立进程是否真正串联；真实供应商验收回答选定模型在实际协议与延迟下是否完成限定任务；隐藏真值后验检查回答某次研究发现与仿真构造是否一致。任何一层都不能替代其余层。

## 当前回归测试

当前运行验收基线是2026-10-29的NOOA `dev473` 升级提交 `358c710`（合并为 `f5b8d96`），在82 x86_64 Linux、CPython 3.12.15上完成。锁文件检查、冻结安装、doctor、Ruff和格式检查通过；全量 **186 passed / 1 skipped**，强制bubblewrap专项 **42 passed**。本次没有新增Mac或ARM回测，也没有把测试目录当作研究服务部署。按维护决定，GitHub Runtime checks已停用，Sphinx文档构建与Pages发布仍启用。

这些是该精确版本的验收快照，不是永久测试总数。历史macOS的50项Python、6项Vue，以及后续104/123项回归记录仍保留在[自审记录](quality.md)，不能冒充当前版本新增平台覆盖。上游弃用和子进程析构警告如实记录，不通过屏蔽警告制造干净输出。

| 测试文件 | 主要覆盖 |
|---|---|
| test_etching.py | 波形可复现、固定recipe、隐藏状态不进观测、物理代理与特征检查 |
| test_research.py | 旧评估器的训练/开发边界、因子合法性、预算与冻结 |
| test_persistent_sessions.py、test_process_cleanup.py | 活对象、函数重定义、语法/运行异常、取消/超时/崩溃、子进程清理 |
| test_code_buffers.py | 精确revision、CAS编辑、虚拟源码追踪、原生工具代理 |
| test_code_experiments.py | 独立实验、参数/依赖、main不污染、stale、篡改拒绝与发布 |
| test_execution_feedback.py、test_execution_correction.py | 有界输出、父子观察进实际下一轮请求、原路径复验、目标检查和只读反思 |
| test_autoresearch.py | 通用代码、会话恢复、失败/超时、讨论限制和worker边界 |
| test_runtime.py | 真实NOOA/SDK请求构造，50k输出、thinking/effort与本地窗口 |
| test_streaming.py | UTF-8增量、工具参数不泄露、附件限定、SSE、幂等、指导和停止 |
| test_capabilities.py | 实际单元格、recipe、run范围、原生Reflexion只读/耗尽和真实判词 |
| test_budget_branches.py | 跨进程预算、真实SDK重试、未知用量和父/分支文件边界 |
| test_orchestration.py | 真实HTTP、NOOA和分支进程，修订、并发、失败/取消隔离、单分支恢复 |
| test_web_recovery.py | 服务接管同一存活worker及陈旧标记处理 |
| test_scientific_stack.py | 实际隔离worker里计算SciPy峰值、sklearn PCA、statsmodels回归、aeon DTW与三维归一化；验证JIT缓存和真实图表落盘 |
| frontend/tests/workspace.test.ts | Vue消息、Unicode重放、身份隔离、指导、审查、预算和图片路径 |

集成测试使用确定性模型服务，但运行真实SDK、HTTP与科学worker/子进程，不能称为“只有mock函数”。同样，它们也不是对真实GLM的随机输出质量保证。Linux后端的运行保证依据具体提交的真实回归；当前用上述82记录，历史Runtime checks只能证明其当时提交。文档CI通过不能解释为运行验收。

扩展分析库后的回归还检查实际NOOA请求上下文包含这些科学库，避免只是把库装在宿主上却没有告诉agent。库栈测试验证代表性计算可执行，不声称所有可选extras和算法都已经验收。

## 原生外层迁移验证

当前迁移分支移除自定义外层执行器，使用标准CodeActV2/SandboxedExecutor。82原生专项目6项通过，验证能力配置、普通异常后活对象持续、官方helper、标准Predict helper声明/调用、本地REST网络允许/拒绝和超时后空状态重建；缓冲区、实验、反思及供应商契约组19项通过。两组范围不同，不相加为完整测试总数。全量与真实模型验收在PR中记录，未完成时不能引用历史186/42当作迁移后结果。

无Landlock场景通过能力对象模拟选择与真实filesystem=off原生运行分别验证；它们不是实际Linux5.0内核测试。原生fork的线程相关弃用警告如实保留。嵌套Predict helper使用确定性客户端测试，不冒称已验证任意付费模型或全部复杂策略组合。

## 真实MiniMax历史纠错限定验收（2026-10-29）

在相同dev473精确提交上，预先用真实隔离worker记录除零失败的 `score` 实验，再让 `MiniMax-M3.1-Flash-Preview` 根据宿主源码与失败观察修改revision，调用 `verify_repair` 重放原checks，并用 `verify_goal` 单独验 `score(2)==4`。宿主实际记录1条原失败修复关系、1项目标通过，未解决失败为0；不是只根据模型回复文字判断。

首次外置验收脚本直接使用裸客户端，漏接正式 `MissionBudget.client`，使NOOA内部 `prompt_cache_key` 进入MiniMax的Anthropic路由并被拒绝；当时模型没有进入纠错步骤。脚本改用正式 `BudgetedClient` 调用链后通过，项目代码不因此放宽供应商参数检查。该失败与修正属于验收入口记录，不应混报为Python回归失败或新版科学能力已失效。

确定性脚本模型验证上下文确实送达；这次真实模型验证选定任务能走通；二者都不能证明复杂公式、任意模型或完整RCA效果会同样成功。真实请求需要私有配置，不能成为公共文档构建依赖。

## 真实GLM历史限定任务验收

一次真实接口验收派生两条独立方向，分别统计公开发现集的目标腔和参考腔run数量。两分支均完成并通过各自审查，实际模型并发峰值为2。第一轮主任务因保守token预留不足而结束，分支结果保留；第二轮复用原分支，完成合并、一次定向修订和PNG/CSV交付。

主报告35条结构化引用通过保存表检查，Predict审查通过并保留五条warning。原生Reflexion实际只得到一次未满意判词，随后预算耗尽；候选自评中的PASS没有被当成原生反思通过。两种状态分别记录。

原始会话和trace包含研究输入，不公开上传；这里公开的是限定任务、口径和结果汇总。它证明并行、状态、预算、校验、修订和流式交付能串联，**没有做传感器根因定位准确率评估**。

## 文档质量的验证门

1. `sphinx-build -n -W`严格构建；任何缺页、无效交叉引用、指令或Mermaid错误都失败。
2. Mermaid在构建期实际渲染为SVG，检查生成数量、SVG XML与viewBox，而非只数代码围栏。
3. 执行实际生成的搜索JS，验证中英文stemmer和中文索引词；浏览器实测中文搜索，不以“脚本文件200”代表功能正确。
4. 检查生成站点内部链接与本地静态资产存在，确认没有私有路径或旧对话被复制进站点。
5. 从AST与JSON生成配置、schema、模块和CLI参考，避免文档手写默认值随代码漂移。
6. 按追踪矩阵核对实现与测试符号；需求未被硬性满足的部分显式标记。
7. 发布后验证GitHub工作流、Pages状态、首页/章节/CSS/SVG，并用浏览器检查实际排版。

这些检查能发现构建与若干事实错误，不能机械判定每一段物理解释都真。最后仍需人工反思：主张是否超出证据、图是否误导因果、数字是否混了统计范围、读者能否从设计走到代码与复现。
