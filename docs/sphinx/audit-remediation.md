# 审核问题修复与验证设计

这次修复以两份审核中的可复现问题为范围。历史物理一致性与案例数值仍保留，不把早期测试通过解释为执行、证据或评估保证。

## 问题到修复的对应

| 问题 | 当前设计 | 对应回归 |
|---|---|---|
| 外层执行边界 | 当前使用NOOA标准fork/proxy与按能力选择的guard；旧白名单/无凭据承诺不再适用，关闭文件限制时依赖部署权限 | test_native_sandbox、test_execution_boundaries |
| worker自报成功、伪造控制帧、超长输出导致串线 | 真实退出码、无控制FD、宿主管理cell顺序、输出限长并排空 | test_execution_boundaries、test_reliability_cleanup |
| 技能缺失或运行配置与声明不一致 | 安装skills registry并将领域正文注入固定上下文；单一策略工厂 | test_capabilities |
| 编造CSV也能成为已支持证据 | 保护执行账本；原始值/模型指标独立重算；显式证据等级 | test_evidence_provenance |
| 反证只有一段文本 | EvidenceCheck区分planned/completed，宿主执行数值predicate并记录结果 | test_evidence_provenance |
| 最后重写改变数字 | 规范Markdown直接分段推送，不做第二次模型重写 | test_reply_writer |
| 自我审查被当成独立审查 | 独立配置模型的只读Predict reviewer；不可用明确not_reviewed | test_orchestration |
| test从未使用、只有一个dev lot | 默认360片、多独立lot、冻结后test评分与机制定位评分 | test_evaluation、test_etching |
| 只报分类满分，无法判断伪相关 | 多故障与无故障对照、共同漂移场景、机制precision/recall及误报/漏报 | run_blind_benchmark、test_evaluation |
| 审查上限和Reflexion失败误标预算耗尽 | revision_limit、not_satisfactory、有限主迭代、真实进展等待 | test_capabilities、test_budget_branches |
| 审计失败计数与finally异常覆盖 | 结构化script退出码，审计失败另存且保留原异常 | test_reliability_cleanup |
| 个人取key脚本、额外内部ID头 | 从公共核心删除；本地配置与共享预算保留 | 代码与发布范围检查 |
| 旧入口混淆、错误噪声注释 | legacy显式命名空间；research默认现代流程；注释明确时间相关噪声 | test_research、test_runtime |
| 开发依赖漏洞、历史CI只构建文档 | 锁定依赖与历史平台/前端工作流；当前Runtime checks停用，82执行运行验收，文档发布保持启用 | 验证章节、npm audit |

## 近期执行与纠错改造

旧的逐cell销毁再反序列化状态无法保留活对象语义，零散报错也不足以支持后续纠偏。PR #10–#12分别补齐长驻内核与失败失效、独立版本实验与产物有效性、持久父子观察与原路径复验；PR #13将NOOA单独升级为dev473。新机制见[执行运行协议](repl-runtime.md)，测试入口为 `test_persistent_sessions.py`、`test_code_buffers.py`、`test_code_experiments.py`、`test_execution_feedback.py`、`test_execution_correction.py`。这些测试和真实MiniMax限定验收证明协议与一个纠错样例，不构成完整RCA准确率保证。

## 留出评估与对照场景

默认`balanced_v3`为360片目标片加每lot一片参考腔室数据；每个时段24个lot，各12个train、6个dev、6个test。recipe、指令波形和步骤时长相同。校准和环境扰动在普通场景中按lot独立抽样。旧90片`legacy_v2`仍可显式生成，保留历史案例可复现性，不用于新的强评估承诺。

六类场景包括背氦/接触、抽气受限、表面状态、边缘环、共同漂移和稳定无故障。中性随机case名与私有映射分开保存。生成场景不等于已经完成六场景agent盲评；缺失报告明确标为report_missing。

收尾评估是用户/宿主操作，不是每次画图都强制执行。先固定ResearchReply和已选公式、模型，再在私有test快照上复用训练态做预测。评分包括balanced accuracy、macro F1、测试样本/lot数、机制precision/recall/F1、误报与漏报、传感器/step定位精度。无正式模型时明确只做机制评分。已冻结定义不能依据test分数反复改写选优。

## 仍需保留的科学限制

哈希、回放和独立重算分别证明来源、可复现性和声明计算的数值一致性，不证明任意研究代码或因果推断正确。探索产物可以用于讨论，但不能仅凭成功退出升级为原始事实。独立配置的审查模型也可能犯错或存在同类偏差。仿真机制评分只验证所构造场景，不等价于实际工厂诊断准确率。

本次实际执行结果及平台覆盖见[验收记录](quality.md)。没有运行的场景、付费模型验收或平台项目不能记为通过。
