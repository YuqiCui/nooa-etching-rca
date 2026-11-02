"""NOOA CodeActV2 scientist with general code execution and durable dialogue."""

from __future__ import annotations

from typing import Annotated

from nooa import Agent, hidden, strategy

from .dataset_models import AcquisitionRequest, DatasetDerivation, DatasetSchema  # noqa: F401 - CodeAct types
from .formula_models import FormulaHandoff, FormulaPlan, FormulaSpec  # noqa: F401 - CodeAct types
from .research_models import (
    CandidateAssessment,
    EvidenceCheck,  # noqa: F401 - public CodeAct type
    EvidenceRef,  # noqa: F401 - public CodeAct type
    Hypothesis,  # noqa: F401 - public CodeAct type
    RawEvidenceSpec,
    ResearchReply,
    ReviewResult,
)

with hidden:
    import json
    import uuid

    from nooa import TextSkill
    from nooa.config.strategy_config import PredictConfig
    from nooa.storage import nosnapshot
    from nooa.strategies import PredictStrategy

    from .capabilities import Capabilities
    from .datasets import DatasetSession
    from .formula_lab import FormulaLab
    from .live_events import MarkdownReplies, artifact_index
    from .live_events import public_artifact_path as artifact_path
    from .local_reflection import reflection_active
    from .reply_writer import StreamingReplyWriter
    from .research_workspace import PythonWorkspace
    from .strategy_factory import reflection_strategy, research_strategy


class AutoresearchAgent(Agent):
    """与领域专家协作的工业过程与多模态数据研究员。根据用户本轮问题选择算法、写Python并依据证据反驳和修订假设。

    专家擅长排除有数学证据的错误方向，不负责替你设计所有实验。重要原则见实际激活的领域/统计技能。
    外层python_cell调用公开方法；所有研究代码在隔离worker运行。先inspect_dataset确认当前范围；子集/父集切换用数据工具。Polars仅为表格处理首选，不是算法限制。
    可自由组合NumPy、SciPy、scikit-learn、aeon、statsmodels、pandas、matplotlib及其他已安装库。
    根据问题选择成熟实现，不因强调Polars就手写替代成熟统计/时序算法。不得访问隐藏真值。
    外层python_cell的cwd/文件系统不属于研究工作目录，不在外层open/read_parquet/savefig。
    把完整源代码交给run_code或code_write/code_edit+code_run（无.py文件的版本化缓冲区）或write_file+run_script；worker的cwd就是workspace；当前只读数据在worker变量DATASET，引用为DATASET_ID。
    PNG等二进制由run_code中的plt.savefig('artifacts/plot.png')直接生成；write_file只接受文本。
    重要数值保存为CSV/Parquet并返回EvidenceRef（id/claim/artifact/column/where/statistic/value/unit/run_ids）。
    value表示where筛出的单行单元格；聚合用mean/count等明确操作。Hypothesis.evidence_ids引用这些ID；
    supported/weak必须有结构化引用。普通探索的表绑定执行记录；确认原始数值用materialize_raw_evidence，正式模型指标用materialize_formula_evidence。
    supported/rejected需要完成并由宿主计算的EvidenceCheck；反证计划不能写成已经执行。summary中的数值也应提供evidence条目。
    可以通过spawn_research自主分派不同问题，通过wait_research/read_branch_result获取结果再综合复核。
    子任务使用同一公开数据、独立工作区和历史；提供任务及必要证据路径，不能假定它读过父会话。
    分支证据可通过branches/<id>/artifacts/...读取和引用。各方向可动态合并、指导或取消。
    review检查后可能进入revise，须定向修复具体问题，不无限重跑探索；审查通过不等于根因已证实。
    使用inspect_candidate可做有界原生Reflexion，它只读已有证据，不发消息、不改研究文件。
    有阶段性观察需要展示时用reply(message_id,brief,image_paths,table_paths)，普通工具参数不推流；短任务可直接返回最终ResearchReply。
    研究进展是阶段性观察；最终交接由宿主在审查完成后发布，research返回结构化ResearchReply即可。
    不复制最终全文多次，不编造数字或路径；记录失败与反证，不凑十个原因。同一数据范围的可序列化变量通过子进程检查点保留；切换范围后重置，文件和历史会保存。
    """

    _workspace: Annotated[PythonWorkspace, hidden, nosnapshot]
    _replies: Annotated[MarkdownReplies | None, hidden, nosnapshot]

    _reply_writer: Annotated[StreamingReplyWriter | None, hidden, nosnapshot]

    _manager: Annotated[object, hidden, nosnapshot]
    _policy: Annotated[Capabilities, hidden, nosnapshot]
    _knowledge_registry: Annotated[object, hidden, nosnapshot]
    _last_verification: Annotated[dict, hidden, nosnapshot]
    _reflection_verdicts: Annotated[list, hidden, nosnapshot]
    _formula_lab: Annotated[object, hidden, nosnapshot]
    _datasets: Annotated[object, hidden, nosnapshot]
    physics: Annotated[TextSkill, nosnapshot]
    statistics: Annotated[TextSkill, nosnapshot]

    @hidden
    def __init__(
        self, workspace, replies=None, reply_writer=None, manager=None, policy=None, **kwargs
    ):
        super().__init__(**kwargs)
        self._workspace = workspace
        self._replies = replies
        self._reply_writer = reply_writer
        self._manager = manager
        self._policy = policy or Capabilities.load()
        self._knowledge_registry = None
        self._last_verification = {}
        self._reflection_verdicts = []
        self._formula_lab = FormulaLab(workspace.run, workspace.events) if workspace else None
        self._datasets = None
        if workspace and any(
            (workspace.run / "input" / n).exists()
            for n in ["runs.parquet", "dataset.json", "samples.parquet", "samples.csv"]
        ):
            self._datasets = DatasetSession(workspace.run, workspace.events)
            workspace.dataset_session = self._datasets
            self._formula_lab.data_session = self._datasets
        from .execution_observer import install_execution_observer

        install_execution_observer(self)
        self.round_number = 0

    async def inspect_dataset(
        self, dataset_ref: str = "", preview_rows: int = 5, include_source: bool = False
    ) -> dict:
        """Inspect a version's modality, schema, identity roles and a small index preview.

        DATASET and DATASET_ID inside run_code/run_script identify the selected
        read-only version. Current scope survives process restarts, Python variables do not.
        Reading a dataset does not switch scope or enable formula comparisons.
        """
        from .dataset_adapter import index_frame

        doc = self._datasets.describe(dataset_ref)
        frame = index_frame(self._datasets.path(doc["id"]), self._datasets.schema(doc["id"]))
        doc["preview"] = json.loads(frame.head(min(20, max(0, preview_rows))).write_json())
        doc["file_count"] = len(doc["files"])
        doc["files"] = dict(list(doc["files"].items())[:30])
        if include_source and doc["transform_path"]:
            source = (self._datasets.path(doc["id"]) / doc["transform_path"]).read_text()
            doc["transform_source"] = source[:32000]
            doc["source_truncated"] = len(source) > 32000
        return doc

    async def list_datasets(self, offset: int = 0, limit: int = 25) -> dict:
        """Page through immutable dataset versions and parent relationships; no scope change."""
        page = self._datasets.registry.list(offset, limit)
        page["datasets"] = [
            {k: d[k] for k in ["id", "title", "rows", "parents", "version", "relation"]}
            | {"modality": d["schema"]["modality"]}
            for d in page["datasets"]
        ]
        return page

    async def read_dataset_lineage(self, dataset_ref: str = "") -> dict:
        """Read the immutable parent graph, transformation provenance and version hashes."""
        self._datasets.authorize(dataset_ref or self._datasets.current()["id"])
        graph = self._datasets.registry.lineage(dataset_ref or self._datasets.current()["id"])
        graph["nodes"] = [
            {
                k: d[k]
                for k in [
                    "id",
                    "title",
                    "rows",
                    "parents",
                    "reason",
                    "content_sha256",
                    "transform_sha256",
                    "params",
                    "provenance",
                ]
            }
            for d in graph["nodes"]
        ]
        return graph

    async def materialize_raw_evidence(self, spec: RawEvidenceSpec) -> dict:
        """Have the host recompute a declared raw value/window/aggregate and save its proof.

        Use for final numerical facts. Free run_code remains available for arbitrary
        exploration. A transformed dataset is marked derived, never a raw measurement.
        Returns a complete EvidenceRef; keep its proof_id and fields unchanged.
        """
        await self._writable()
        from .trusted_evidence import materialize_raw

        return materialize_raw(self._workspace.run, spec, self._workspace.events)

    async def materialize_formula_evidence(
        self, study_id: str, trial_id: str, metric: str, evidence_id: str, claim: str
    ) -> dict:
        """Recompute a canonical formula trial metric from its saved predictions."""
        await self._writable()
        from .trusted_evidence import materialize_formula

        return materialize_formula(
            self._workspace.run, self._formula_lab, study_id, trial_id, metric, evidence_id, claim
        )

    async def derive_dataset(self, derivation: DatasetDerivation, select: bool = True) -> dict:
        """Register a dataset produced by ANY Python code you wrote and executed.

        Write a source file and output directory in your workspace; output contains
        an index and optional assets/steps. Declare data_schema, parent references,
        parameters, reason and relation (subset/transform/merge). A subset may inherit
        immutable asset paths from parents. Parents are never overwritten. Do not
        fake labels or held-out samples. select=True switches this agent to the child.
        """
        await self._writable()
        async with self._workspace.lock:
            doc = self._datasets.derive(derivation)
            if select:
                await self._workspace.close()
                self._datasets.select(doc["id"])
                from .research_context import refresh_domain_context

                refresh_domain_context(self)
            return doc

    async def select_dataset(self, dataset_ref: str) -> dict:
        """Select an existing immutable version for this agent and push the previous scope.

        Subsequent run_code uses DATASET; formula plans freeze this version. Other
        agents retain their own scopes. Live Python variables reset on a scope switch.
        """
        await self._writable()
        async with self._workspace.lock:
            await self._workspace.close()
            result = self._datasets.select(dataset_ref)
            from .research_context import refresh_domain_context

            refresh_domain_context(self)
            return result

    async def return_dataset(self, target: str = "parent") -> dict:
        """Return to the previous scope, or target='root' for this agent's bound root."""
        await self._writable()
        if target not in {"parent", "root"}:
            raise ValueError("Choose parent or root")
        async with self._workspace.lock:
            await self._workspace.close()
            result = self._datasets.select(back=target)
            from .research_context import refresh_domain_context

            refresh_domain_context(self)
            return result

    async def request_data(self, request: AcquisitionRequest) -> dict:
        """Request additional entries through the future unified acquisition interface.

        The default provider is UNAVAILABLE. This release has no network or KG
        connector. Preserve source/criteria/reason/parent/lead references and request_id;
        never claim data was fetched unless status='supplied' returns a DatasetRef.
        """
        await self._writable()
        return await self._datasets.acquire(request)

    @hidden
    async def _writable(self):
        if reflection_active.get():
            raise PermissionError(
                "Local reflection is read-only; no research execution, writes, replies or task mutations"
            )
        if not getattr(self._workspace, "research_enabled", True):
            raise PermissionError("Discussion mode cannot execute research or modify tasks")

    async def run_code(self, code: str, timeout_s: int = 120) -> dict:
        """Execute ANY scientific Python source in your persistent work directory.

        Imports, function definitions, files, plots and subprocess scripts are all
        supported. Print concise results; full output and source are journaled.
        Returns ok, output (text), error and full_output (record path); no stdout key.
        DATASET is the selected read-only pathlib.Path, DATASET_ID its DatasetRef.
        timeout_s is a hard worker deadline, capped at 300 seconds. On timeout the
        worker and its children are killed; files remain, live variables reset.
        """
        await self._writable()
        if not getattr(self._workspace, "research_enabled", True):
            raise PermissionError(
                "Discussion mode: use existing artifacts and reply; research execution is disabled"
            )
        return await self._workspace.execute(code, timeout_s)

    async def create_formula_study(self, plan: FormulaPlan) -> dict:
        """Freeze a pairwise formula comparison: public train/dev, lot folds, metric, models.

        Use for expert-proposed custom operators and multi-round formula selection.
        All versions in one study share the same data, folds, model parameters and ranking.
        Changing labels/metric/models/data requires a new study. No test runs are used.
        """
        await self._writable()
        return self._formula_lab.create(plan)

    async def begin_formula_comparison(self, study_id: str) -> dict:
        """Explicitly resume an existing formula comparison for THIS turn.

        Use when asked to continue comparison or confirm the best formula. It enables
        the selection/incomplete handoff requirement without changing the saved study.
        Simply reading results, plotting or ordinary exploration does not need this tool.
        The requirement resets next turn; historical studies never force new experiments.
        """
        await self._writable()
        return self._formula_lab.begin(study_id)

    async def register_formula(self, study_id: str, formula: FormulaSpec) -> dict:
        """Snapshot an arbitrary Python raw-trace operator as an immutable formula version.

        For kind=per_sample, write extract(sample, steps, params) -> scalar or None.
        A sample is a Polars frame for tables/time series, a Path for image/text/mixed assets.
        For kind=learned define fit(samples, steps, params, labels=None) and
        transform(state, sample, steps, params). fit sees only fold training inputs/labels;
        transform sees one sample and learned state. All code remains arbitrary Python.
        trace contains only time_s, step_id and declared sensors/steps; steps contains
        step_id/start_s/end_s. No labels are supplied. Implement windows and cross-sensor
        fusion freely in Python. Supply the exact human expression, units and parameters.
        Re-registering a name requires change_reason and creates v2, preserving v1; each version is a required singleton
        comparison until explicitly retired. mean(a*b) is different from mean(a)*mean(b).
        """
        await self._writable()
        return self._formula_lab.register(study_id, formula)

    async def evaluate_formula_set(self, study_id: str, formula_refs: list[str]) -> dict:
        """Evaluate one formula or a combination of registered versions against ALL planned models.

        The selected DatasetRef is frozen; source is recomputed with per-sample isolation
        or learned fit/transform inside each training fold. Preprocessing is fitted within each training
        fold, dev predictions and host-calculated metrics are saved. Failures remain visible;
        retry resumes missing models, not completed work. Never supply invented score tables.
        """
        await self._writable()
        return await self._formula_lab.evaluate(study_id, formula_refs)

    async def read_formula_study(self, study_id: str = "") -> dict:
        """Read durable formula versions, frozen protocols, failed trials and selections across rounds/branches."""
        if study_id:
            return self._formula_lab.public(self._formula_lab.get(study_id))
        return {"studies": self._formula_lab.catalog()}

    async def retire_formula_set(self, study_id: str, set_id: str, reason: str) -> dict:
        """Explicitly exclude a scientifically invalid/failed candidate with an auditable reason.

        This changes the comparison universe and invalidates the previous selection.
        Do not exclude a legitimate candidate merely to improve the winning score.
        """
        await self._writable()
        return self._formula_lab.retire(study_id, set_id, reason)

    async def freeze_formula_selection(self, study_id: str) -> dict:
        """Select best among tried formulas only after complete, equal-budget comparisons.

        Recompute winning features/model and require identical predictions and scores.
        Return its study_id/status/selection_id in ResearchReply.formula_comparison.
        If blocked, return status='incomplete' with an honest reason instead of a winner.
        New versions invalidate the selection; previous selections remain in history.
        """
        await self._writable()
        return await self._formula_lab.freeze(study_id)

    async def code_write(self, name: str, source: str, expected_revision: int = 0) -> dict:
        """Create/replace a named code buffer without executing or creating a .py file.

        Creation requires expected_revision=0; replacement requires the current
        revision from code_read. Revisions persist in the supervisor database.
        Edits never change already-running definitions: explicitly code_run next.
        """
        await self._writable()
        if not self._workspace.research_enabled:
            raise PermissionError("Discussion mode cannot modify code buffers")
        from .code_buffers import CodeBuffers

        return CodeBuffers(self._workspace.run).write(name, source, expected_revision)

    async def code_edit(self, name: str, expected_revision: int, old: str, new: str) -> dict:
        """Replace one exact nonempty match atomically in a named buffer.

        Stale revision or absent/ambiguous text fails without modifying history.
        Editing does not execute code or update existing Python function objects.
        """
        await self._writable()
        if not self._workspace.research_enabled:
            raise PermissionError("Discussion mode cannot modify code buffers")
        from .code_buffers import CodeBuffers

        return CodeBuffers(self._workspace.run).write(name, "", expected_revision, old=old, new=new)

    async def code_read(
        self, name: str, revision: int = 0, offset: int = 0, limit: int = 12000
    ) -> dict:
        """Read exact source text by character page; revision=0 selects latest.

        Returns source_ref/hash and next_offset, including for very long lines.
        Use the returned explicit revision for code_edit and code_run.
        """
        from .code_buffers import CodeBuffers

        doc = CodeBuffers(self._workspace.run).get(name, revision)
        text = doc.pop("source")
        offset, limit = max(0, offset), min(32000, max(1, limit))
        return {
            **doc,
            "content": text[offset : offset + limit],
            "characters": len(text),
            "offset": offset,
            "start_line": text[:offset].count("\n") + 1,
            "next_offset": offset + limit if len(text) > offset + limit else None,
        }

    async def code_list(self) -> list[dict]:
        """List durable named code buffers and latest revisions."""
        from .code_buffers import CodeBuffers

        return CodeBuffers(self._workspace.run).catalog()

    async def code_run(self, name: str, revision: int, timeout_s: int = 120) -> dict:
        """Execute exactly one immutable buffer revision in the scientific namespace.

        revision must be positive. No automatic replay: existing old references
        stay old, deleted names remain until explicitly removed or namespace reset.
        Execution history identifies the exact source revision via source_ref.
        """
        await self._writable()
        if revision < 1:
            raise ValueError("Choose an explicit positive revision")
        from .code_buffers import CodeBuffers

        doc = CodeBuffers(self._workspace.run).get(name, revision)
        return await self._workspace.execute(doc["source"], timeout_s, source_ref=doc["source_ref"])

    async def code_experiment(
        self,
        name: str,
        revision: int,
        checks: str,
        dependencies: dict[str, int] | None = None,
        parameters: dict | None = None,
        timeout_s: int = 120,
    ) -> dict:
        """Run explicit source revisions in a fresh disposable scientific worker.

        Dependencies execute in supplied insertion order, then candidate, then checks.
        No main REPL objects/files are inherited. DATASET is a frozen input copy;
        PARAMETERS contains JSON inputs. Use assert-based checks for the goal and
        original failure path. Success means these supplied checks passed, not that
        the research goal is independently proven. Artifacts remain staged; use
        code_publish only after inspecting code_read_experiment. Main memory survives
        any experiment failure. No automatic replay or publication.
        """
        await self._writable()
        if not self._workspace.research_enabled:
            raise PermissionError("Discussion mode cannot run experiments")
        from .code_experiments import CodeExperiments

        return await CodeExperiments(self._workspace).execute(
            name, revision, dependencies or {}, parameters or {}, checks, timeout_s
        )

    async def code_read_experiment(self, experiment_id: str) -> dict:
        """Read the host manifest, exact sources/parameters/checks, output and validity.

        Source/data edits make historical results stale even after publication.
        Failed, cancelled and timed-out experiments are retained for diagnosis.
        """
        from .code_experiments import CodeExperiments

        return CodeExperiments(self._workspace).get(experiment_id)

    async def code_publish(self, experiment_id: str) -> dict:
        """Publish current checked experiment artifacts through the host.

        Immutable destination artifacts/<experiment_id> cannot replace earlier
        evidence. Refuses failed/stale experiments or changed staged artifacts.
        Published files do not activate definitions in the main REPL.
        """
        await self._writable()
        if not self._workspace.research_enabled:
            raise PermissionError("Discussion mode cannot publish experiments")
        from .code_experiments import CodeExperiments

        async with self._workspace.lock:
            return CodeExperiments(self._workspace).publish(experiment_id)

    async def read_observation(
        self, observation_id: str, offset: int = 0, limit: int = 12000
    ) -> dict:
        """Read paginated host execution evidence, including exact source, error and parent ID."""
        return self._workspace.observations.read(observation_id, offset, limit)

    async def verify_repair(
        self,
        failure_id: str,
        name: str,
        revision: int,
        dependencies: dict[str, int] | None = None,
        parameters: dict | None = None,
        timeout_s: int = 120,
    ) -> dict:
        """Verify a correction against the exact original failed path in a fresh worker.

        For failed experiments the host reuses original checks and JSON parameters;
        source names/order must match and at least one revision must change. For a
        scientific cell the entire original source is the replayed check path after
        explicit new definitions. Self-contained inline errors may need a separate
        reproduction experiment; this tool never edits away the failing check.
        Inputs must match the original frozen bytes/scope. Unrelated successful
        executions cannot resolve failures. Success verifies this path, not all goals.
        """
        await self._writable()
        from .code_experiments import CodeExperiments

        workspace = self._workspace
        journal = workspace.observations
        failure = journal.get(failure_id)
        if failure["status"] != "failed":
            raise ValueError("Choose an actual failed observation")
        expected_inputs = failure.get("input_files")
        expected_dataset = failure.get("dataset_ref")
        if failure["kind"] == "experiment":
            original = CodeExperiments(workspace).get(failure["experiment_id"])
            if [*(dependencies or {}), name] != [src["name"] for src in original["sources"]]:
                raise ValueError("Repair must preserve the original source dependency names/order")
            if parameters is not None and parameters != original["parameters"]:
                raise ValueError("Repair must reuse original parameters")
            revisions = {**(dependencies or {}), name: revision}
            if all(revisions[src["name"]] == src["revision"] for src in original["sources"]):
                raise ValueError("Edit a source revision before verifying repair")
            checks, parameters = original["checks"], original["parameters"]
            expected_inputs, expected_dataset = original["input_files"], original["dataset_ref"]
        elif failure["kind"] == "scientific":
            checks = failure["code"]
        else:
            raise ValueError("Choose a scientific or experiment failure")
        verified = await self.code_experiment(
            name, revision, checks, dependencies, parameters, timeout_s
        )
        passed = (
            verified["status"] == "checked"
            and verified["validity"] == "current"
            and verified["input_files"] == expected_inputs
            and verified["dataset_ref"] == expected_dataset
        )
        identity = journal.record(
            "repair",
            "passed" if passed else "failed",
            repairs=failure_id,
            experiment_id=verified["experiment_id"],
            checks=checks,
            output="\n".join(r["output"] for r in verified["executions"]),
        )
        if passed:
            journal.resolve(failure_id, identity)
        return {
            "observation_id": identity,
            "status": "verified" if passed else "unverified",
            "failure_id": failure_id,
            "experiment_id": verified["experiment_id"],
            "goal_satisfied": None,
        }

    async def verify_goal(
        self, goal: str, experiment_id: str, checks: str, timeout_s: int = 120
    ) -> dict:
        """Separately check a stated goal criterion by recomputing a current experiment.

        Reuses exact sources/inputs/parameters; checks must assert the named criterion.
        Records only this supplied criterion as passed, not independent truth of the
        entire research task. Does not resolve prior failures or publish artifacts.
        """
        await self._writable()
        from .code_experiments import CodeExperiments

        original = CodeExperiments(self._workspace).get(experiment_id)
        if not goal.strip() or original["status"] != "checked" or original["validity"] != "current":
            raise ValueError("Choose a named goal and a current checked experiment")
        candidate = original["sources"][-1]
        dependencies = {src["name"]: src["revision"] for src in original["sources"][:-1]}
        result = await self.code_experiment(
            candidate["name"],
            candidate["revision"],
            checks,
            dependencies,
            original["parameters"],
            timeout_s,
        )
        passed = (
            result["status"] == "checked"
            and result["validity"] == "current"
            and result["input_files"] == original["input_files"]
            and result["dataset_ref"] == original["dataset_ref"]
        )
        identity = self._workspace.observations.record(
            "goal",
            "passed" if passed else "failed",
            goal=goal,
            checks=checks,
            experiment_id=result["experiment_id"],
            output="\n".join(r["output"] for r in result["executions"]),
        )
        return {
            "observation_id": identity,
            "goal": goal,
            "supplied_criterion_passed": passed,
            "experiment_id": result["experiment_id"],
        }

    async def write_file(self, path: str, content: str) -> dict:
        """Save ANY text/source file within the research workspace, without running it.

        A convenience for long scripts: avoids nested code-string quoting. Use
        run_script to execute a saved script with checked exit status; run_code for exploration.
        """
        await self._writable()
        if not getattr(self._workspace, "research_enabled", True):
            raise PermissionError("Discussion mode cannot modify research files")
        target = (self._workspace.work / path).resolve()
        if not target.is_relative_to(self._workspace.work):
            raise ValueError("Choose a path within the research workspace")
        code = (
            "from pathlib import Path\n"
            f"_target = Path({str(target)!r})\n"
            "_target.parent.mkdir(parents=True, exist_ok=True)\n"
            f"print('saved', _target.name, _target.write_text({content!r}), 'characters')"
        )
        return await self._workspace.execute(code)

    async def run_script(
        self, path: str, args: list[str] | None = None, timeout_s: int = 120
    ) -> dict:
        """Run a saved Python script with live stdout/stderr and checked exit status.

        path must resolve inside workspace. args are passed literally, without a
        shell. A nonzero script exit returns ok=false; inspect the saved traceback.
        Long-lived worker variables are separate from the script process namespace.
        DATASET and DATASET_ID are injected into scripts too; no path argv is needed.
        """
        await self._writable()
        if not getattr(self._workspace, "research_enabled", True):
            raise PermissionError("Discussion mode cannot run scripts")
        target = (self._workspace.work / path).resolve()
        if not target.is_relative_to(self._workspace.work) or target.suffix != ".py":
            raise ValueError("Choose a Python script inside the research workspace")
        return await self._workspace.execute(
            f"run_script({path!r}, args={args or []!r})", timeout_s, script=target, args=args
        )

    async def reply(
        self,
        message_id: str,
        brief: str,
        image_paths: list[str] | None = None,
        table_paths: list[str] | None = None,
    ) -> dict:
        """Publish your exact Markdown through the canonical SSE channel.

        Provide clear reply points and exact verified evidence in brief, including
        numbers/units/limitations to preserve. The host delivers the exact text without another model rewriting it;
        the result includes the delivered Markdown.
        Choose your own stable message_id (e.g. progress-1); it is a publication key,
        not an incoming user-message ID. Reuse it for an intentional retry/update.
        No literal-string convention, argument preview or tool_stream is needed.
        Only existing artifacts/ PNG/JPEG/WebP/GIF and CSV files may be attached.
        Ordinary tools execute with complete arguments and stream only real output.
        """
        if reflection_active.get():
            raise PermissionError("Local reflection cannot publish replies")
        if self._replies is None:
            raise RuntimeError("Reply channel is unavailable")
        if self._reply_writer is not None:
            return await self._reply_writer.send(message_id, brief, image_paths, table_paths)
        return self._replies.send(message_id, brief, image_paths, table_paths)

    async def list_executions(self, offset: int = 0, limit: int = 20) -> list[dict]:
        """List recorded executions, newest first; namespaces reset on a new research turn."""
        return self._workspace.list_executions(offset, limit)

    async def read_execution(self, execution_ref: str, offset: int = 0, limit: int = 12000) -> dict:
        """Read immutable executed source, stdout/stderr and status without running Python.

        Each text field is independently paged by character offset/limit. Collected
        output may be truncated; original code and retained diagnostics remain available.
        """
        return self._workspace.read_execution(execution_ref, offset, limit)

    async def list_artifacts(self, prefix: str = "artifacts") -> list[dict]:
        """List actual artifact paths, types and sizes before referencing a file."""
        return artifact_index(self._workspace.run, prefix)

    async def read_artifact(self, path: str, max_chars: int = 12000) -> dict:
        """Read existing UTF-8 Markdown/JSON/CSV/text/source, without executing code."""
        target = artifact_path(self._workspace.run, path)
        if target.suffix.lower() not in {".md", ".json", ".csv", ".txt", ".py"}:
            raise ValueError("Choose a text artifact; images can be attached with reply")
        if target.stat().st_size > 8_000_000:
            raise ValueError("File too large; use research code to select a smaller evidence table")
        content = target.read_text()
        cap = min(32000, max(100, max_chars))
        return {"path": path, "content": content[:cap], "truncated": len(content) > cap}

    @strategy(research_strategy(Capabilities()))
    async def research(self, message: str) -> ResearchReply:
        """Continue autonomous research using the full conversation and new input.

        Choose and implement your own experiments. Test expert feedback rather
        than mechanically accepting it, revise hypotheses, persist code/evidence,
        and return a checkpoint with questions the expert can efficiently judge.
        """
        ...

    @strategy(PredictStrategy(config=PredictConfig(max_retries=1)))
    async def discuss(self, message: str) -> ResearchReply:
        """Discuss existing evidence using the full history, without running code.

        Do not introduce newly computed numbers. Clarify, challenge or interpret
        prior findings; distinguish proposed future experiments from completed work.
        """
        ...

    @strategy(research_strategy(Capabilities()))
    async def chat(self, message: str) -> ResearchReply:
        """用中文讨论已有证据。可读取已有artifact并调用reply实时回复。

        本模式不允许研究代码执行或修改文件。只解释已完成结果，不生成新计算值。
        所有给用户看的文字用reply发送；最后return_result只用于保存结构化交接。
        """
        ...

    @strategy(research_strategy(Capabilities()))
    async def revise(self, message: str, previous: ResearchReply, review: dict) -> ResearchReply:
        """针对证据检查或审查指出的具体问题补证并修订，不重复无关实验。

        保留原证据和失败记录，返回完整修订后的ResearchReply与EvidenceRef；预算不足时明确降级结论。
        通过审查只表示检查通过，不表示已证明具体部件根因。
        """
        ...

    @strategy(PredictStrategy(config=PredictConfig(max_retries=1)))
    async def review(self, evidence_bundle: dict) -> ReviewResult:
        """独立审查给定报告与已复算的证据包，返回结构化问题单，不运行研究代码。

        检查闭环重复证据、设定值与step均值、lot伪重复、小样本置换下界、选择泄漏、
        OPLS变换与恢复比例、以及关联越界到部件根因。每个问题须指向具体claim_id/证据ID，
        写明反证和可执行检查；没有证据支持的疑问应列为warning而非捏造blocking问题。
        claim_id只填写一个现存hypothesis/evidence ID或summary/scope；多个引用放evidence_ids数组。
        不得把review自身格式错误当作研究报告的问题；文字里以标点列举多个ID本身不构成科学错误。
        若材料不够复核，应说明缺口，不能假装重新读取文件或重算过。只审查当前报告，不重写历史。
        """
        ...

    @strategy(reflection_strategy(Capabilities()))
    async def _inspect_candidate(
        self, candidate: ResearchReply, execution_evidence: dict
    ) -> CandidateAssessment:
        """只读审查候选结果，检查证据一致性、反证和观测边界，返回评估及建议。
        可读现有artifact，不执行研究代码、不写文件、不发消息或派生任务。不是根因认证。
        """
        ...

    async def inspect_candidate(self, candidate: ResearchReply) -> dict:
        """Run bounded native experimental Reflexion over existing candidate evidence.

        No research execution, file mutations, publishing or task mutations are allowed
        during the local reflection. The last native verdict is returned explicitly;
        budget exhaustion is never silently labelled as a passed review.
        """
        await self._writable()
        if not self._policy.reflexion_enabled:
            raise PermissionError("Local Reflexion is disabled in capabilities policy")
        candidate = ResearchReply.model_validate(candidate)
        self._reflection_verdicts = []
        token = reflection_active.set(True)
        result = None
        interrupted = None
        try:
            result = await self._inspect_candidate(
                candidate,
                execution_evidence=self._workspace.execution_context(),
                _strategy=reflection_strategy(self._policy),
            )
        except BaseException as error:  # noqa: BLE001 - persist verdicts on cancellation, then re-raise
            interrupted = error
        finally:
            reflection_active.reset(token)
        accepted = bool(
            self._reflection_verdicts and self._reflection_verdicts[-1]["is_satisfactory"]
        )
        budget_stopped = (interrupted is None or isinstance(interrupted, Exception)) and getattr(
            self._llm, "budget_exhausted", False
        )
        record = {
            "status": "interrupted"
            if interrupted and not budget_stopped
            else "passed"
            if accepted and not interrupted and not budget_stopped
            else "budget_exhausted"
            if budget_stopped
            else "not_satisfactory",
            "assessment": result.model_dump() if result else None,
            "verdicts": self._reflection_verdicts,
            "experimental": True,
            "reason": "shared_model_budget"
            if budget_stopped
            else type(interrupted).__name__
            if interrupted
            else "native_verdict",
        }
        folder = self._workspace.run / "reflections"
        folder.mkdir(exist_ok=True)
        (folder / (uuid.uuid4().hex + ".json")).write_text(
            json.dumps(record, ensure_ascii=False, indent=2)
        )
        if self._workspace.events:
            self._workspace.events.emit(
                "reflection.end", status=record["status"], reason=record["reason"]
            )
        if interrupted and record["status"] == "interrupted":
            raise interrupted
        return record

    async def spawn_research(
        self,
        task: str,
        title: str = "",
        evidence_paths: list[str] | None = None,
        dataset_ref: str = "",
    ) -> dict:
        """Start an independent CodeActV2 researcher; give a focused question and optional evidence files.
        Returns an agent_id immediately; the branch owns its code, workspace, history,
        summary and review. Use wait_research for results; repeated identical live tasks are deduplicated.
        """
        await self._writable()
        if self._manager is None:
            raise RuntimeError("Research coordinator is unavailable")
        return await self._manager.spawn(task, title, evidence_paths or [], dataset_ref or None)

    async def list_research(self) -> list[dict]:
        """List persisted research branches and actual lifecycle/review status."""
        return self._manager.list() if self._manager else []

    async def wait_research(
        self, agent_ids: list[str] | None = None, timeout_s: int = 30
    ) -> list[dict]:
        """Wait up to timeout_s for a result; does not cancel branches on timeout.
        Returns completed results and current statuses. Read final evidence and review
        before accepting a branch's conclusions. Can be called again while jobs run.
        """
        if self._manager is None:
            return []
        return await self._manager.wait(agent_ids, timeout_s)

    async def read_branch_result(self, agent_id: str) -> dict:
        """Read a completed branch's report, review and verified artifact paths."""
        if self._manager is None:
            raise RuntimeError("Research coordinator is unavailable")
        return self._manager.result(agent_id)

    async def guide_research(self, agent_id: str, message: str) -> dict:
        """Send expert guidance to one active branch; it enters that agent's native event channel."""
        await self._writable()
        return self._manager.guide(agent_id, message)

    async def cancel_research(self, agent_id: str) -> dict:
        """Cancel one branch and clean up its workers, retaining evidence and events."""
        await self._writable()
        return await self._manager.cancel(agent_id)

    async def resume_research(self, agent_id: str, message: str) -> dict:
        """Resume a saved, inactive branch using its history/files and the shared mission budget."""
        await self._writable()
        return await self._manager.resume(agent_id, message)
