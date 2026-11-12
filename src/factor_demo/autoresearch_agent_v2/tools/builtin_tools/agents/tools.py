"""向通用会话开放授权范围内的并行委派工具。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Annotated, Any, ClassVar

from pydantic import Field, StringConstraints

if TYPE_CHECKING:
    from ....agent import Session
    from ....runtime import AgentRuntime
    from ...base import ToolFactory


class AgentTools:
    """绑定当前会话身份，不接受模型指定父会话或工作目录。"""

    namespace: ClassVar[str] = "agent"
    tool_names: ClassVar[tuple[str, ...]] = (
        "agent__spawn",
        "agent__status",
        "agent__wait",
        "agent__cancel",
        "agent__read_result",
    )

    def __init__(self, runtime: AgentRuntime, session: Session) -> None:
        """保存宿主创建的运行时与当前调用者身份。"""
        self.runtime = runtime
        self.session = session
        self._active = True

    @classmethod
    @asynccontextmanager
    async def bind(cls, session: Session) -> AsyncIterator[AgentTools]:
        """从宿主绑定取得运行时，不创建第二个调度器。"""
        if session.runtime is None:
            raise RuntimeError("AgentTools 需要宿主绑定 AgentRuntime")
        instance = cls(session.runtime, session)
        try:
            yield instance
        finally:
            instance._active = False

    def register_tools(self, factory: ToolFactory) -> None:
        """登记通用委派和结果查询接口。"""
        definitions = (
            ("agent__spawn", self.spawn, "启动独立子会话；立即返回，允许多个子任务并行。"),
            ("agent__status", self.status, "查询当前会话直接委派的子任务状态。"),
            ("agent__wait", self.wait, "等待至少一个子结果；结果也会自动进入后续上下文。"),
            ("agent__cancel", self.cancel, "取消当前会话的子任务及其后代并清理资源。"),
            ("agent__read_result", self.read_result, "按结果标识分页读取当前会话收到的全文。"),
        )
        for name, callback, description in definitions:
            factory.register_callable(name, callback, description=description)
        factory.add_instructions(
            "子会话使用相同基础规则，通过任务正文和所选 skill 分工。"
            "子会话 finish 的结果自动投递当前会话，无需反复查询。"
            "任务格式和结果要求由用户请求及 skill 决定。"
        )

    async def spawn(
        self,
        task: str,
        description: Annotated[
            str,
            StringConstraints(strip_whitespace=True, min_length=1),
            Field(description="简要说明调用子 Agent 的目的，以及该子任务如何帮助完成当前任务。"),
        ],
        skills: list[str] | None = None,
        allow_tools: str | list[str] = "all",
    ) -> dict[str, Any]:
        """按调用目的、任务正文和所选 skill 提交独立委派。

        Args:
            task: 交给子 Agent 执行的具体任务，包含必要的输入和结果要求。
            description: 简要说明调用子 Agent 的目的。
            skills: 子会话启动时加载的技能标识。
            allow_tools: 子会话可以使用的工具，不能超过父会话权限。

        Returns:
            子会话标识、调用目的和当前状态。
        """
        self._require_active()
        return await self.runtime.spawn(
            self.session, task, skills, allow_tools, description=description,
        )

    def status(
        self,
        agent_id: str | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> dict[str, Any]:
        """查询当前会话的直接子委派。"""
        self._require_active()
        return self.runtime.status(self.session, agent_id, offset, limit)

    async def wait(self) -> dict[str, Any]:
        """等待结果而不消费父收件箱。"""
        self._require_active()
        return await self.runtime.wait(self.session)

    async def cancel(self, agent_id: str) -> dict[str, Any]:
        """取消获准管理的子委派。"""
        self._require_active()
        return await self.runtime.cancel(self.session, agent_id)

    def read_result(self, result_id: str, offset: int = 0, limit: int = 4_000) -> dict[str, Any]:
        """读取收到的完整结果，不开放运行时数据库路径。"""
        self._require_active()
        return self.runtime.read_result(self.session, result_id, offset, limit)

    def _require_active(self) -> None:
        """会话退出后撤销旧工具实例和已借出方法。"""
        if not self._active:
            raise RuntimeError("AgentTools 已关闭")
