"""将宿主批准的技能目录绑定到独立会话。"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, ClassVar

from ....skills import SkillContext, SkillRegistry
from ...base import ToolFactory

if TYPE_CHECKING:
    from ....agent.session import Session


class SkillTools:
    """提供技能发现、显式加载和受限引用读取。"""

    namespace: ClassVar[str] = "skill"
    tool_names: ClassVar[tuple[str, ...]] = (
        "skill__list", "skill__load", "skill__read_resource",
    )

    def __init__(self, session: Session, context: SkillContext) -> None:
        """保存会话和技能状态，不创建执行能力。"""
        self._session = session
        self._context = context
        self._active = True

    @classmethod
    def register(
        cls, session: Session, *, registry: SkillRegistry, preload: Sequence[str] = ()
    ) -> None:
        """登记技能资源；预加载不依赖模型主动调用加载工具。"""
        selected = tuple(preload)

        @asynccontextmanager
        async def create_context(current: Session) -> AsyncIterator[SkillContext]:
            """根据有效工具集合恢复并预加载本会话技能。"""
            context = SkillContext(
                registry,
                available_tools=current.tool_policy.allowed_names,
                snapshots=current.state.skill_snapshots,
            )
            context.preload(selected)
            current.state.skill_snapshots = context.snapshots()
            yield context

        session.register_resource(SkillContext, create_context)
        session.register_stateful_tool(cls)

    @classmethod
    @asynccontextmanager
    async def bind(cls, session: Session) -> AsyncIterator[SkillTools]:
        """借用已创建的技能上下文，并在退出后撤销工具引用。"""
        instance = cls(session, session.resource(SkillContext))
        try:
            yield instance
        finally:
            instance._active = False

    def register_tools(self, factory: ToolFactory) -> None:
        """登记不会扩大工具授权或执行脚本的技能操作。"""
        factory.register_callable(
            "skill__list", self.list_skills,
            description="查看宿主批准的技能目录与当前会话加载状态。",
        )
        factory.register_callable(
            "skill__load", self.load,
            description="在当前会话加载指定技能正文；不会开放额外工具或权限。",
        )
        factory.register_callable(
            "skill__read_resource", self.read_resource,
            description="读取已加载技能快照中的参考文件；路径必须在该技能内。",
        )

    def _require_active(self) -> None:
        """拒绝关闭后使用旧工具引用。"""
        if not self._active:
            raise RuntimeError("技能工具已经关闭")

    def list_skills(self) -> list[dict[str, Any]]:
        """返回技能元数据，不加载正文。"""
        self._require_active()
        return self._context.catalog()

    def load(self, skill_id: str) -> dict[str, Any]:
        """激活技能并同步可恢复会话状态。"""
        self._require_active()
        snapshot = self._context.load(skill_id)
        self._session.state.skill_snapshots = self._context.snapshots()
        return {
            "skill_id": skill_id, "version": snapshot.description.version,
            "sha256": snapshot.sha256, "loaded": True,
        }

    def read_resource(self, skill_id: str, path: str) -> str:
        """返回指定快照资源的完整文本。"""
        self._require_active()
        return self._context.read_resource(skill_id, path)
