"""将四个工作项工具绑定到宿主授权的研究计划存储。"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel, ValidationError

from ....database import create_task_plan_service
from ....database.services import TaskPlanService
from ...base import ToolFactory
from ...call_context import execution_id
from .typing import TaskCreateInput, TaskGetInput, TaskListInput, TaskUpdateInput

if TYPE_CHECKING:
    from ....agent.session import Session


class TaskTools:
    """显式登记存储根目录，模型只能操作当前研究任务中的工作项。"""

    tool_names: ClassVar[tuple[str, ...]] = ("task_create", "task_get", "task_update", "task_list")
    """启动前可检查的公开工具名称。"""

    namespace: ClassVar[str] = "task"
    """会话内工具组标识；公开工具名固定使用 task_ 前缀。"""

    def __init__(self, service: TaskPlanService, session_id: str, actor_id: str):
        """绑定固定范围和宿主身份。

        Args:
            service: 当前研究任务的存储服务。
            session_id: 用于隔离不同会话中的调用标识。
            actor_id: 写入审计记录的 agent 标识。
        """
        self._service = service
        self._session_id = session_id
        self._actor_id = actor_id
        self._active = True

    @classmethod
    def register(cls, session: "Session", *, storage_root: Path) -> None:
        """登记持久化资源与工具，存储必须在模型工作目录之外。

        Args:
            session: 尚未启动的会话。
            storage_root: 宿主专用存储目录，恢复时再次指定同一目录。

        Raises:
            ValueError: 存储目录位于模型工作目录内。
        """
        root = storage_root.expanduser().resolve()
        workdir = session.state.workdir.expanduser().resolve()
        if root.is_relative_to(workdir):
            raise ValueError("研究计划存储必须位于模型工作目录之外")
        research_task_id = session.state.task_id

        @asynccontextmanager
        async def create_service(current: "Session") -> AsyncIterator[TaskPlanService]:
            """按登记时的研究范围创建服务，退出时释放连接资源。"""
            if current.state.task_id != research_task_id:
                raise ValueError("登记后不能改变研究任务标识")
            if root.is_relative_to(current.state.workdir.expanduser().resolve()):
                raise ValueError("研究计划存储必须位于模型工作目录之外")
            service = create_task_plan_service(root, research_task_id)
            try:
                yield service
            finally:
                service.close()

        session.register_resource(TaskPlanService, create_service, eager=False)
        session.register_stateful_tool(cls)

    @classmethod
    @asynccontextmanager
    async def bind(cls, session: "Session") -> AsyncIterator["TaskTools"]:
        """取得宿主登记的存储，并在会话退出后撤销工具访问。

        Args:
            session: 提供研究存储和调用身份的会话。

        Yields:
            当前会话专属的工作项工具。
        """
        service = await session.bind_resource(TaskPlanService, _require_registered_service)
        instance = cls(service, session.state.session_id, session.state.agent_id)
        try:
            yield instance
        finally:
            instance._active = False

    def register_tools(self, factory: ToolFactory) -> None:
        """由输入模型生成 Schema，并统一处理参数错误。

        Args:
            factory: 当前会话工具注册表。
        """
        definitions = (
            ("create", TaskCreateInput, "创建研究工作项，返回稳定 ID，初始状态为 pending。"),
            ("get", TaskGetInput, "按 task_id 读取完整工作项；不存在或已删除时返回 null。"),
            (
                "update",
                TaskUpdateInput,
                (
                    "仅更新提交字段；metadata 顶层合并，null 删除键；"
                    "status=deleted 删除条目且不能同时修改其他字段。"
                ),
            ),
            ("list", TaskListInput, "按创建顺序列出当前研究任务的未删除工作项摘要。"),
        )
        for operation, model, description in definitions:
            factory.register(
                f"task_{operation}",
                self._make_handler(operation, model),
                description=description,
                parameters=model.model_json_schema(),
            )
        factory.add_instructions(
            "复杂研究可用 task_create 分解工作，task_list 查看计划，task_get 获取详情。"
            "实质进展或计划变化时用 task_update 更新；未提交字段保持不变。"
            "完成状态可重新打开，多项可以同时进行。completed 仅记录完成声明，不验证结论。"
            "简单任务不必建立清单。metadata 可保存结果摘要和产物路径，但不自动验证文件。"
        )

    def _make_handler(self, operation: str, model: type[BaseModel]):
        """构造保留缺省字段语义的工具处理器。"""

        def invoke(**arguments):
            """校验整个请求后执行，避免部分更新。"""
            if not self._active:
                raise RuntimeError("工作项工具已经关闭")
            try:
                request = model.model_validate(arguments)
            except (ValidationError, ValueError):
                return {"error": "invalid_arguments", "hint": "请检查字段名称、类型和空值规则。"}
            if operation == "update" and request.model_fields_set == {"task_id"}:
                return {"error": "empty_update", "hint": "请至少提交一个需要修改的字段。"}
            call_id = execution_id.get()
            identity = f"{len(self._session_id)}:{self._session_id}:{call_id}" if call_id else None
            return self._service.execute(
                operation,
                request.model_dump(mode="json", exclude_unset=operation == "update"),
                actor_id=self._actor_id,
                execution_id=identity,
            )

        return invoke


def _require_registered_service(session):
    """没有宿主配置时拒绝创建默认存储，避免猜测授权位置。"""
    raise RuntimeError("TaskTools 必须先通过 register 配置宿主存储")
