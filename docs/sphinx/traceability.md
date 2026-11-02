# 需求到代码、测试与限制的追踪

矩阵以当前公共仓库中的文件为可检查对象。测试名称按文件列出以减少符号重命名造成的假链接；完整函数可在[源码生成索引](_generated/modules.md)查看。历史真实模型与科学结论的证据等级见[验证章节](validation.md)。

| ID | 实现入口 | 自动验证 | 限制与结论 |
|---|---|---|---|
| R01 | etch_model.py、etch_features.py、etch_sensors.py | test_etching.py | 已实现波形和特征；物理系数未生产标定 |
| R02 | etch_recipe.py、etch_validation.py | test_etching.py | 指令哈希与波形一致；反馈可变 |
| R03 | autoresearch_agent.py、research_workspace.py | test_autoresearch.py、test_orchestration.py | 通用代码可用；受限执行后端为macOS与Linux |
| R04 | research_models.py、reply_writer.py、Vue组件 | test_reply_writer.py、workspace.test.ts | 中文+图表已实现；逐次改算完整报告字段待补 |
| R05 | autoresearch.py、web.py、LiveEvents | test_streaming.py、test_orchestration.py | 指导后续生效；不改写正在生成中的同一请求 |
| R06 | evidence.py | test_capabilities.py | 表值、范围与引用可检；不证明任意变换正确 |
| R07 | review_pipeline.py | test_orchestration.py | 有界修订和原始review保留；Predict不独立重算 |
| R08 | local_reflection.py、inspect_candidate | test_capabilities.py | 原生组合与实际判词；只读局部实验 |
| R09 | research_branches.py、branch_worker.py | test_budget_branches.py、test_orchestration.py | 独立工作区与进程；主研究直接派生，非递归分布式 |
| R10 | budget.py、runtime.py | test_budget_branches.py、test_orchestration.py | 单mission共享；不是全部署账户级额度 |
| R11 | provider_stream.py、live_events.py、useResearch.ts | test_streaming.py、workspace.test.ts | 真流、续传、去重；写请求ID不覆盖所有科学副作用 |
| R12 | rounds修订、SQLite快照、web恢复 | test_web_recovery.py、test_orchestration.py | 文件和事件可恢复；内存变量不恢复、非exactly-once |
| R13 | docs/sphinx、docs.yml、check_docs.py | 严格构建与站点检查 | 公开文档与源仓；不是线上研究服务 |
| R14 | formula_models.py、formula_lab.py、formula_runner.py、FormulaStudies.vue | test_formula_lab.py、formulas.test.ts | 任意Python标量特征、四类固定模型与强制交接；不是任意模型、因果证明或独立最终测试 |
| R15 | datasets.py、dataset_adapter.py、dataset_api.py、DatasetPanel.vue | test_datasets.py、datasets.test.ts | 版本/血缘/嵌套/恢复/分支与本地导入；实际外部拉数和KG未接入 |
| R16 | feature_execution.py、feature_runner.py、artifact_store.py、review_pipeline.py | test_learned_features.py、test_multimodal_features.py、test_audit_regressions.py | 逐对象/折内隔离与可恢复交付；任意代码语义与因果仍需核验 |
| R17 | persistent_kernel.py、session_cell.py、native_sandbox.py、code_buffers.py | test_persistent_sessions.py、test_native_sandbox.py、test_code_buffers.py | 活对象只在当前生命周期；科学编译错误保留、运行失败失效；外层遵循原生异常，编辑不执行 |
| R18 | code_experiments.py、execution_provenance.py | test_code_experiments.py | 精确依赖/参数、失败隔离、显式发布与stale拒绝；不推断任意对象依赖 |
| R19 | execution_observations.py、autoresearch_agent.py、research_context.py | test_execution_correction.py、test_execution_feedback.py | 父子证据、原路径复验和具名目标分开；反思只读，不认证任意科学结论 |

## 八项NOOA能力的具体组合

| 原生能力 | 应用补齐部分 | 公开实现文件 |
|---|---|---|
| LoopGuardConfig | 事件展示、正常长等待语义 | strategy_factory.py、provider_stream.py |
| pre/postconditions | 公开范围、recipe、数值引用与活跃子任务规则 | evidence.py |
| Context与TextSkill | 领域规则、反馈、当前任务与有界假设 | research_context.py、knowledge/ |
| PredictStrategy | 审查证据包、问题schema与格式修复 | review_pipeline.py、research_models.py |
| ReflexionStrategy | 只读副作用限制、V2上下文桥接、实际判词与耗尽状态 | local_reflection.py、autoresearch_agent.py |
| QueueManager/Channel | 跨进程任务注册、结果与指导来源 | research_branches.py |
| AdmissionControl/Broker | token预留/估算、跨进程连接与所有调用覆盖 | budget.py |
| 独立Agent与Storage | 工作区、host、快照、恢复及前端分支身份 | branch_worker.py、web.py、useResearch.ts |

## 变更后如何避免追踪失效

配置参考与模块列表每次构建从AST/JSON重新生成。`check_docs.py`验证本矩阵列出的Python模块和测试文件存在、所有章节进入导航、Mermaid数量与SVG一致，并检查站点本地链接。它不会理解所有自然语言主张；涉及状态机、预算、故障设定和算法能力的修改仍须人工逐项复核矩阵与正文。

如果添加跨平台worker、跨案例知识或严格报告schema，不能只更新首页“支持”的文字；应同时补需求状态、实现入口、可执行验收、失败边界和文档自审记录。
