"""为通用子会话复制定义与授权，不复制父会话的活资源。"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING

from ..agent import Session
from ..agent.model import Model
from ..python_tools import PythonTools
from ..sandboxing import Sandbox
from ..tools import FileTools, NotebookTools, SandboxTools

if TYPE_CHECKING:
    from .typing import ChildRequest


class SessionFactory:
    """使用共同的会话内核创建独立子会话，应用补充宿主资源装配。"""

    def __init__(
        self,
        *,
        configure: Callable[[Session], None] | None = None,
        model_factory: Callable[[Session, ChildRequest], Model] | None = None,
    ) -> None:
        """保存显式装配函数，不创建模型连接或执行资源。"""
        self._configure = configure
        self._model_factory = model_factory

    def __call__(self, parent: Session, request: ChildRequest) -> Session:
        """复制工具与技能定义，为子会话配置独立写目录及授权输入。"""
        model = (
            self._model_factory(parent, request)
            if self._model_factory is not None
            else Model(parent.loop.model.config)
        )
        child = Session(
            model,
            request.state,
            system_prompt=parent.loop.system_prompt,
            history_config=parent.loop.history_config,
            tool_registry=parent.tool_registry,
            allow_tools=request.allow_tools,
            skill_registry=parent.skill_registry,
            skills=request.skills,
        )
        child.state.datasets = parent.state.datasets.copy()
        self._copy_builtin_declarations(parent, child)
        try:
            sandbox = parent.resource(Sandbox)
        except RuntimeError:
            pass
        else:
            config = sandbox.config.model_copy(update={"workdir": child.state.workdir})
            child.register_resource(
                Sandbox,
                partial(SandboxTools.create_sandbox, config=config),
                eager=False,
            )
        if self._configure is not None:
            self._configure(child)
        return child

    @staticmethod
    def _copy_builtin_declarations(parent: Session, child: Session) -> None:
        """兼容旧式内置工具登记，带宿主范围的扩展仍由 configure 显式装配。"""
        independent = {FileTools, NotebookTools, PythonTools, SandboxTools}
        configured = set(child.stateful_tool_types())
        for tool_type in parent.stateful_tool_types():
            if tool_type in independent and tool_type not in configured:
                child.register_stateful_tool(tool_type)
