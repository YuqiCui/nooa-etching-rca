"""将通用 JSON 文档操作绑定到宿主固定的任务与所有者。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import JsonValue

from ...base import ToolFactory
from .store import RecordStore

if TYPE_CHECKING:
    from ....agent.session import Session


def _require_registered_store(session: Session) -> AbstractAsyncContextManager[RecordStore]:
    """缺少宿主显式登记时失败，不从会话路径推导存储授权。"""
    raise RuntimeError("请先通过 RecordTools.register 登记文档存储")


class RecordTools:
    """不固定业务字段，只提供所属会话的读取与版本条件写入。"""

    namespace: ClassVar[str] = "record"
    tool_names: ClassVar[tuple[str, ...]] = ("record__get", "record__put")

    def __init__(self, store: RecordStore, scope: str, owner: str) -> None:
        """固定宿主身份，模型不能切换所有者。"""
        self._store = store
        self._scope = scope
        self._owner = owner
        self._active = True

    @classmethod
    def register(cls, session: Session, *, storage_root: Path) -> None:
        """登记位于模型可写目录外的宿主文档存储。"""
        root = storage_root.expanduser().resolve()
        workdir = session.state.workdir.expanduser().resolve()
        if root.is_relative_to(workdir):
            raise ValueError("JSON 文档存储必须位于模型工作目录之外")
        scope = session.state.task_id
        owner = session.state.agent_id

        @asynccontextmanager
        async def create_store(current: Session) -> AsyncIterator[RecordStore]:
            """在会话启动时再次核对固定范围和目录边界。"""
            if current.state.task_id != scope or current.state.agent_id != owner:
                raise ValueError("登记后不能改变文档所有者或任务范围")
            if root.is_relative_to(current.state.workdir.expanduser().resolve()):
                raise ValueError("JSON 文档存储必须位于模型工作目录之外")
            store = RecordStore(root)
            try:
                yield store
            finally:
                store.close()

        session.register_resource(RecordStore, create_store, eager=False)
        session.register_stateful_tool(cls)

    @classmethod
    @asynccontextmanager
    async def bind(cls, session: Session) -> AsyncIterator[RecordTools]:
        """绑定当前所有者，并在退出时关闭工具访问。"""
        store = await session.bind_resource(RecordStore, _require_registered_store)
        instance = cls(store, session.state.task_id, session.state.agent_id)
        try:
            yield instance
        finally:
            instance._active = False

    def register_tools(self, factory: ToolFactory) -> None:
        """登记通用文档工具，不添加研究流程或验收要求。"""
        factory.register_callable(
            "record__get", self.get,
            description="读取当前 agent 的 JSON 文档及版本；不存在时 document=null。",
            recover_validation_errors=True,
        )
        factory.register_callable(
            "record__put", self.put,
            description="按 expected_revision 完整替换 JSON 文档；0 表示仅创建新文档。",
            recover_validation_errors=True,
        )

    def _require_active(self) -> None:
        """拒绝关闭后继续使用旧工具引用。"""
        if not self._active:
            raise RuntimeError("文档工具已经关闭")

    def get(self, document_id: str) -> dict[str, Any]:
        """读取当前任务与所有者下的文档。"""
        self._require_active()
        return self._store.get(self._scope, self._owner, document_id)

    def put(
        self, document_id: str, content: JsonValue, expected_revision: int
    ) -> dict[str, Any]:
        """条件写入任意合法 JSON 内容，不推断业务完成状态。"""
        self._require_active()
        return self._store.put(
            self._scope, self._owner, document_id, content, expected_revision
        )
