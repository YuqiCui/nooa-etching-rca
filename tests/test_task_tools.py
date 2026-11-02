"""工作项真实持久化、参数边界和会话执行回归。"""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select

from factor_demo.autoresearch_agent_v2.agent import Session, SessionState
from factor_demo.autoresearch_agent_v2.agent.model import Model
from factor_demo.autoresearch_agent_v2.config import ModelConfig
from factor_demo.autoresearch_agent_v2.database import create_task_plan_service, tables
from factor_demo.autoresearch_agent_v2.database.sqlite.runtime import SQLiteDatabase
from factor_demo.autoresearch_agent_v2.tools import TaskTools, ToolFactory


@pytest.fixture
def task_factory(tmp_path):
    """提供真实数据库和工具调用入口。"""
    service = create_task_plan_service(tmp_path, "research")
    tools = TaskTools(service, "session", "main")
    factory = ToolFactory()
    tools.register_tools(factory)
    yield factory
    service.close()


def call(factory, name, **arguments):
    """在同步测试中调用真实工具。"""
    return asyncio.run(factory.call(f"task_{name}", arguments))


def test_crud_metadata_and_tombstones(task_factory):
    """增删改查、缺省字段、重开和元数据合并遵循接口约定。"""
    factory = task_factory
    assert call(factory, "list") == {"tasks": []}
    created = call(
        factory, "create", subject="  检查数据  ", metadata={"keep": 1, "nested": {"old": 1}}
    )
    assert created == {"task": {"id": "T1", "subject": "检查数据", "status": "pending"}}
    original = call(factory, "get", task_id="T1")["task"]
    assert original["description"] == ""
    assert original["active_form"] is None
    result = call(
        factory,
        "update",
        task_id="T1",
        status="completed",
        active_form="正在检查",
        metadata={"nested": {"new": 2}, "new": [1]},
    )
    assert result["status_change"] == {"from": "pending", "to": "completed"}
    assert call(factory, "get", task_id="T1")["task"]["metadata"] == {
        "keep": 1,
        "nested": {"new": 2},
        "new": [1],
    }
    call(
        factory,
        "update",
        task_id="T1",
        status="in_progress",
        active_form=None,
        metadata={"keep": None},
    )
    current = call(factory, "get", task_id="T1")["task"]
    assert current["active_form"] is None
    assert "keep" not in current["metadata"]
    assert current["created_at"] == original["created_at"]
    assert call(factory, "update", task_id="T1", metadata={})["updated_fields"] == []
    assert call(factory, "get", task_id="T1")["task"] == current
    call(factory, "create", subject="第二项")
    assert [item["id"] for item in call(factory, "list")["tasks"]] == ["T1", "T2"]
    call(factory, "update", task_id="T1", status="deleted")
    assert call(factory, "get", task_id="T1") == {"task": None}
    assert len(call(factory, "list")["tasks"]) == 1
    assert call(factory, "update", task_id="T1", status="deleted")["updated_fields"] == []
    assert call(factory, "update", task_id="T1", status="pending")["error"] == "task_deleted"
    assert call(factory, "create", subject="第三项")["task"]["id"] == "T3"


@pytest.mark.parametrize(
    "operation,arguments",
    [
        ("create", {"subject": " "}),
        ("create", {"subject": 42}),
        ("create", {"subject": "x", "status": "completed"}),
        ("create", {"subject": "x", "metadata": {"x": float("nan")}}),
        ("create", {"subject": "x", "metadata": {"x": None}}),
        ("update", {"task_id": "T1", "status": "blocked"}),
        ("update", {"task_id": "T1", "subject": "changed", "description": None}),
        ("update", {"task_id": "T1", "status": "deleted", "subject": "changed"}),
        ("update", {"task_id": "T1", "active_form": " "}),
        ("update", {"task_id": "T1", "metadata": None}),
        ("get", {"task_id": "T1", "research_task_id": "other"}),
        ("list", {"status": "pending"}),
    ],
)
def test_invalid_inputs_do_not_partially_write(task_factory, operation, arguments):
    """整次校验失败不得改变已有记录。"""
    call(task_factory, "create", subject="original")
    before = call(task_factory, "get", task_id="T1")
    assert call(task_factory, operation, **arguments)["error"] == "invalid_arguments"
    assert call(task_factory, "get", task_id="T1") == before
    assert len(call(task_factory, "list")["tasks"]) == 1


def test_missing_and_empty_update(task_factory):
    """未知记录与空补丁返回可恢复错误。"""
    assert call(task_factory, "get", task_id="T99") == {"task": None}
    assert (
        call(task_factory, "update", task_id="T99", status="deleted")["error"] == "task_not_found"
    )
    assert call(task_factory, "update", task_id="T99")["error"] == "empty_update"


def make_session(tmp_path, identity="s", research="r"):
    """构造不访问真实模型的独立会话。"""
    workdir = tmp_path / "workspace"
    workdir.mkdir(exist_ok=True)
    state = SessionState(session_id=identity, agent_id="main", task_id=research, workdir=workdir)
    model = Model(ModelConfig(model="openai/fixture", api_key="fixture"))
    session = Session(model, state)
    TaskTools.register(session, storage_root=tmp_path / "host")
    return session


def test_session_restore_scope_replay_and_close(tmp_path):
    """跨会话恢复、任务隔离和执行去重使用真实数据库。"""

    async def scenario():
        session = make_session(tmp_path)
        async with session:
            result = await session.call("task_create", {"subject": "first"}, call_id="call-1")
            borrowed = session.loop.tools
        with pytest.raises(RuntimeError):
            await borrowed.call("task_list", {})
        async with make_session(tmp_path) as restored:
            assert (
                await restored.call("task_create", {"subject": "first"}, call_id="call-1") == result
            )
            conflict = await restored.call(
                "task_create", {"subject": "different"}, call_id="call-1"
            )
            assert conflict["error"] == "execution_conflict"
            assert len((await restored.call("task_list", {}))["tasks"]) == 1
        async with make_session(tmp_path, identity="s2") as other_session:
            assert len((await other_session.call("task_list", {}))["tasks"]) == 1
            second = await other_session.call("task_create", {"subject": "first"}, call_id="call-1")
            assert second["task"]["id"] == "T2"
        async with make_session(tmp_path, research="other") as other_research:
            assert await other_research.call("task_get", {"task_id": "T1"}) == {"task": None}

    asyncio.run(scenario())


def test_loop_exposes_and_runs_task_tools(tmp_path):
    """模拟模型通过原生协议创建工作项；存储操作均真实执行。"""
    session = make_session(tmp_path)
    responses = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": identity,
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            ],
        }
        for identity, name, arguments in [
            ("create-1", "task_create", {"subject": "check"}),
            ("finish-1", "finish", {"content": "done"}),
        ]
    ]
    session.loop.model.completion = AsyncMock(side_effect=responses)

    async def scenario():
        async with session:
            assert await session.run("检查数据") == "done"
            assert (await session.call("task_list", {}))["tasks"][0]["subject"] == "check"
            schemas = {item["function"]["name"] for item in session.schemas()}
            assert {"task_create", "task_get", "task_update", "task_list"} <= schemas

    asyncio.run(scenario())


def test_storage_is_outside_workspace(tmp_path):
    """宿主不能误把计划放入模型可写目录。"""
    session = make_session(tmp_path)
    with pytest.raises(ValueError, match="工作目录之外"):
        TaskTools.register(session, storage_root=tmp_path / "workspace" / "plans")


def test_concurrent_patches_and_audit(tmp_path):
    """多个连接同时修改不同元数据键时不丢失更新，审计与结果一致。"""
    service = create_task_plan_service(tmp_path, "r")
    factory = ToolFactory()
    TaskTools(service, "s", "main").register_tools(factory)
    call(factory, "create", subject="shared")

    def update(index):
        store = create_task_plan_service(tmp_path, "r")
        try:
            return store.execute(
                "update",
                {"task_id": "T1", "metadata": {str(index): index}},
                actor_id="main",
                execution_id=f"patch-{index}",
            )
        finally:
            store.close()

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(result["success"] for result in pool.map(update, range(12)))
    assert call(factory, "get", task_id="T1")["task"]["metadata"] == {
        str(index): index for index in range(12)
    }
    service.close()
    database = SQLiteDatabase(tmp_path / "task_plan.sqlite", tables.task_plan)
    with database.transaction() as connection:
        rows = connection.execute(select(tables.task_changes)).mappings().all()
        assert len(rows) == 13
        assert all(row["actor_id"] == "main" for row in rows)
    database.close()


def test_failed_audit_rolls_back_create_and_receipt(tmp_path, monkeypatch):
    """审计写入失败时任务与执行收据一起回滚，重试可以正常创建。"""
    service = create_task_plan_service(tmp_path, "r")
    arguments = {"subject": "first", "description": "", "active_form": None, "metadata": {}}
    original = service._record_change

    def fail(*args):
        """模拟事务中途失败。"""
        raise RuntimeError("模拟写入失败")

    monkeypatch.setattr(service, "_record_change", fail)
    with pytest.raises(RuntimeError, match="模拟写入失败"):
        service.execute("create", arguments, actor_id="main", execution_id="once")
    assert service.execute("list", {}, actor_id="main") == {"tasks": []}
    monkeypatch.setattr(service, "_record_change", original)
    result = service.execute("create", arguments, actor_id="main", execution_id="once")
    assert result["task"]["id"] == "T1"
    assert service.execute("create", arguments, actor_id="main", execution_id="once") == result
    service.close()
