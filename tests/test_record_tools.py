"""验证通用 JSON 文档的版本事务、所有者隔离和会话绑定。"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from factor_demo.autoresearch_agent_v2.agent import Session, SessionState
from factor_demo.autoresearch_agent_v2.agent.model import Model
from factor_demo.autoresearch_agent_v2.config import ModelConfig
from factor_demo.autoresearch_agent_v2.tools import ToolFactory
from factor_demo.autoresearch_agent_v2.tools.builtin_tools.records import RecordStore, RecordTools


def test_json_documents_are_schema_agnostic_versioned_and_persistent(tmp_path):
    """文档支持任意 JSON 根值，条件替换不会覆盖不同版本。"""
    store = RecordStore(tmp_path)
    assert store.get("task", "main", "research.json") == {"document": None}
    assert store.put("task", "main", "research.json", {"自由字段": [1, None]}, 0) == {
        "document_id": "research.json", "revision": 1,
    }
    assert store.put("task", "main", "research.json", "不应覆盖", 0) == {
        "error": "revision_conflict", "current_revision": 1,
    }
    store.close()
    restored = RecordStore(tmp_path)
    assert restored.get("task", "main", "research.json")["document"]["content"] == {
        "自由字段": [1, None],
    }
    assert restored.put("task", "main", "research.json", None, 1)["revision"] == 2
    assert restored.get("task", "main", "research.json")["document"]["content"] is None


def test_owners_and_task_scopes_cannot_read_or_replace_each_other(tmp_path):
    """相同文档名在任务和 agent 边界内独立保存。"""
    store = RecordStore(tmp_path)
    store.put("task", "parent", "state", {"parent": True}, 0)
    assert store.get("task", "child", "state") == {"document": None}
    assert store.get("other", "parent", "state") == {"document": None}
    store.put("task", "child", "state", {"child": True}, 0)
    assert store.get("task", "parent", "state")["document"]["content"] == {"parent": True}


def test_concurrent_compare_and_replace_has_one_winner(tmp_path):
    """两个真实连接竞争同一版本时，只能有一个更新成功。"""
    first, second = RecordStore(tmp_path), RecordStore(tmp_path)
    first.put("task", "main", "state", {}, 0)
    ready = Barrier(2)

    def replace(store, value):
        """同步进入竞争点后，执行各自独立的写事务。"""
        ready.wait(timeout=5)
        return store.put("task", "main", "state", value, 1)

    with ThreadPoolExecutor(2) as pool:
        jobs = [pool.submit(replace, first, 1), pool.submit(replace, second, 2)]
        results = [job.result(timeout=10) for job in jobs]
    assert sum("error" not in result for result in results) == 1
    assert sum(result.get("error") == "revision_conflict" for result in results) == 1
    assert first.get("task", "main", "state")["document"]["revision"] == 2


def test_closed_store_nonfinite_values_and_invalid_versions_are_rejected(tmp_path):
    """非法内容和版本在写入之前失败，关闭的实例不可再使用。"""
    store = RecordStore(tmp_path)
    with pytest.raises(ValueError):
        store.put("task", "main", "state", float("nan"), 0)
    with pytest.raises(ValueError):
        store.put("task", "main", "state", ("不是 JSON 数组",), 0)
    with pytest.raises(ValueError):
        store.put("task", "main", "state", {1: "不是 JSON 对象键"}, 0)
    with pytest.raises(ValueError):
        store.put("task", "main", "state", {}, True)
    assert store.get("task", "main", "state") == {"document": None}
    store.close()
    with pytest.raises(RuntimeError, match="关闭"):
        store.get("task", "main", "state")


def test_tool_owner_cannot_be_supplied_by_model(tmp_path):
    """工具参数不接受 owner 字段；合法调用始终使用宿主绑定身份。"""
    store = RecordStore(tmp_path)
    tools = RecordTools(store, "task", "main")
    factory = ToolFactory()
    tools.register_tools(factory)
    result = asyncio.run(factory.call("record__put", {
        "document_id": "state", "content": {}, "expected_revision": 0, "owner": "other",
    }))
    assert result["error"] == "invalid_argument"
    assert store.get("task", "other", "state") == {"document": None}


def test_session_binding_persists_and_revokes_borrowed_tools(tmp_path):
    """会话退出关闭旧引用，重建会话可读取已提交文档。"""
    workdir = tmp_path / "workspace"
    workdir.mkdir()

    def session():
        """构建不调用真实模型的工具会话。"""
        model = Model(ModelConfig(model="openai/fixture", api_key="fixture"))
        current = Session(model, SessionState(
            session_id="session", agent_id="main", task_id="task", workdir=workdir,
        ))
        RecordTools.register(current, storage_root=tmp_path / "storage")
        return current

    async def scenario():
        """只执行真实持久化工具，不启动模型或执行沙箱。"""
        async with session() as first:
            borrowed = first.resource(RecordTools)
            await first.call("record__put", {
                "document_id": "state", "content": ["任意字段"], "expected_revision": 0,
            })
        with pytest.raises(RuntimeError, match="关闭"):
            borrowed.get("state")
        async with session() as second:
            result = await second.call("record__get", {"document_id": "state"})
            assert result["document"]["content"] == ["任意字段"]

    asyncio.run(scenario())


def test_model_writable_record_storage_and_symlink_database_are_rejected(tmp_path):
    """文档存储不能放入模型目录，也不能通过数据库链接访问其他文件。"""
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    model = Model(ModelConfig(model="openai/fixture", api_key="fixture"))
    session = Session(model, SessionState(
        session_id="session", agent_id="main", task_id="task", workdir=workdir,
    ))
    with pytest.raises(ValueError, match="之外"):
        RecordTools.register(session, storage_root=workdir / "records")
    root = tmp_path / "store"
    root.mkdir()
    (root / "records.sqlite3").symlink_to(tmp_path / "secret")
    with pytest.raises(ValueError, match="符号链接"):
        RecordStore(root)


def test_denied_record_tools_do_not_create_storage(tmp_path):
    """禁用整组文档工具时，不创建数据库或绑定存储资源。"""
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    storage = tmp_path / "storage"
    session = Session(
        Model(ModelConfig(model="openai/fixture", api_key="fixture")),
        SessionState(session_id="s", agent_id="main", task_id="t", workdir=workdir),
        allow_tools=["finish"],
    )
    RecordTools.register(session, storage_root=storage)

    async def scenario():
        """检查真实启动后的资源状态和宿主目录副作用。"""
        async with session:
            with pytest.raises(RuntimeError, match="not ready"):
                session.resource(RecordStore)
            assert not storage.exists()

    asyncio.run(scenario())
