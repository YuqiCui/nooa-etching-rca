"""按用户确认表检查实际请求、直接调用及恢复权限。"""

import asyncio
import json
from pathlib import Path

import pytest
from test_agent_runtime import (
    FixtureModel,
    assign,
    collect,
    delivery,
    finished,
    new_worker,
    root_session,
)
from test_autoresearch_web import call, configure
from test_capabilities import public_case

from factor_demo.autoresearch_agent_v2.agent import Session
from factor_demo.autoresearch_agent_v2.agent.model import Model
from factor_demo.autoresearch_agent_v2.errors import UnknownToolError
from factor_demo.autoresearch_agent_v2.python_tools import PythonTools
from factor_demo.autoresearch_agent_v2.runtime import AgentRuntime, SessionFactory
from factor_demo.autoresearch_agent_v2.sandboxing import Sandbox
from factor_demo.autoresearch_agent_v2.tools import TaskTools
from factor_demo.autoresearch_agent_v2.tools.role_policy import role_tool_names
from factor_demo.autoresearch_web import research_turn
from factor_demo.live_events import LiveEvents

# 预期清单来自确认表，不从实现中的白名单推导。
COMMON = {
    "StartWorker", "AssignTask", "CheckWorkerStatus", "WaitWorkers", "CollectWorkerResult",
    "read_file", "write_file", "edit_file", "apply_patch", "register_dataset", "report", "finish",
}
COMPUTATION = {
    "sandbox__run_shell", "python__describe_class", "Statistics.summarize",
    "Statistics.pearson_correlation", "notebook__create_notebook", "notebook__list_notebooks",
    "notebook__write_new_cell", "notebook__write_cells", "notebook__list_cells",
    "notebook__read_cell", "notebook__edit_cell", "notebook__exec_cell", "notebook__run_cells",
}
DISABLED = {
    "skill__list", "skill__load", "skill__read_resource",
    "record__get", "record__put", "report_progress",
    "task_create", "task_get", "task_update", "task_list",
}
ROUTER_SKILLS = {"skill__list", "skill__load", "skill__read_resource"}


def test_actual_web_router_and_worker_match_approved_table(tmp_path, monkeypatch):
    """模型可见能力、直接调用和资源绑定均匹配角色，Worker 仍可获得执行沙箱。"""
    configure(monkeypatch)
    run = public_case(tmp_path / "case")
    sessions = {}
    original_start = Session.start
    root_calls = 0
    worker_calls = 0
    worker_id = None

    async def capture_start(session):
        result = await original_start(session)
        sessions[session.state.agent_id] = session
        if session.state.parent_agent_id is None:
            with pytest.raises(RuntimeError, match="not ready"):
                session.resource(Sandbox)
            with pytest.raises(RuntimeError, match="not ready"):
                session.resource(PythonTools)
        else:
            config = session.resource(Sandbox).config
            assert config.workdir == session.state.workdir
            dataset = Path(session.state.datasets["DATASET"].path).resolve()
            assert dataset.is_relative_to(config.workdir) or any(
                dataset.is_relative_to(path) for path in config.read_paths
            )
        assert TaskTools not in session.stateful_tool_types()
        return result

    async def completion(self, messages, *, tools=None):
        nonlocal root_calls, worker_calls, worker_id
        session = sessions[self.state.agent_id]
        worker = self.state.parent_agent_id is not None
        expected = COMMON | (COMPUTATION if worker else ROUTER_SKILLS)
        assert set(session.tool_policy.allowed_names) == expected
        assert {tool["function"]["name"] for tool in tools} == {
            name for name in expected if not name.startswith("Statistics.")
        }
        forbidden = DISABLED if worker else (DISABLED - ROUTER_SKILLS) | COMPUTATION
        for name in forbidden:
            with pytest.raises((UnknownToolError, PermissionError)):
                await session.loop.tools.call(name, {})
        if worker:
            assert "feature-research" not in messages[0]["content"]
            assert not self.state.skill_snapshots
            worker_calls += 1
            if worker_calls == 1:
                return {"role": "assistant", "tool_calls": [
                    call("record__put", {"document_id": "forbidden", "content": {},
                                         "expected_revision": 0}, "blocked-record"),
                    call("task_create", {"subject": "forbidden"}, "blocked-task"),
                    call("report_progress", {"content": "forbidden"}, "blocked-report"),
                ]}
            assert all("UnknownToolError" in message["content"] for message in messages[-3:])
            return {"role": "assistant", "tool_calls": [call("finish", {"content": delivery()}, "finish")]}
        root_calls += 1
        assert root_calls <= 7
        if root_calls == 1:
            assert not self.state.skill_snapshots
            assert "# 已加载技能" not in messages[0]["content"]
            calls = [call("skill__load", {"skill_id": "feature-research"}, "load-skill"),
                     call("sandbox__run_shell", {"command": "echo forbidden"}, "blocked-shell"),
                     call("task_create", {"subject": "forbidden"}, "blocked-task"),
                     call("report_progress", {"content": "forbidden"}, "blocked-report")]
        elif root_calls == 2:
            assert "# 任务分发与管理" not in messages[0]["content"]
            assert any(message["role"] == "tool" and "# 任务分发与管理" in message["content"]
                       for message in messages)
            assert len(self.state.skill_snapshots) == 1
            assert all("UnknownToolError" in message["content"] for message in messages[-3:])
            calls = [call("StartWorker", {"worker_type": "general"}, "start")]
        elif root_calls == 3:
            worker_id = json.loads(messages[-1]["content"])["worker_id"]
            calls = [call("AssignTask", {"worker_id": worker_id, "task": "检查证据",
                                         "deliver_list": ["结果"]}, "assign")]
        elif root_calls == 4:
            calls = [call("WaitWorkers", {}, "wait")]
        elif root_calls == 5:
            calls = [call("CollectWorkerResult", {"worker_id": worker_id}, "collect")]
        else:
            return {"role": "assistant", "tool_calls": [call("finish", {"content": "完成"}, "finish")]}
        return {"role": "assistant", "tool_calls": calls}

    monkeypatch.setattr(Session, "start", capture_start)
    monkeypatch.setattr(Model, "completion", completion)
    events = LiveEvents(run, "role-policy")
    try:
        asyncio.run(research_turn(run, "fixture", "研究任务", "research", events=events))
        assert worker_calls == 2
        assert not (run / ".autoresearch/tasks").exists()
        assert not any(row["type"] == "reply.commit" and row["data"]["publication"] == "progress"
                       for row in events.since(0))
    finally:
        events.close()


def test_router_can_write_edit_and_patch_files(tmp_path, monkeypatch):
    """保留用户新增开启的三项文件修改能力，验证实际文件效果。"""
    configure(monkeypatch)
    run = public_case(tmp_path / "case")
    responses = iter([
        call("write_file", {"path": "artifacts/research_notes.md", "content": "draft\n"}, "write"),
        call("read_file", {"path": "artifacts/research_notes.md"}, "read"),
        call("edit_file", {"path": "artifacts/research_notes.md", "old_string": "draft",
                           "new_string": "reviewed"}, "edit"),
        call("read_file", {"path": "artifacts/research_notes.md"}, "reread"),
        call("apply_patch", {"input": "*** Begin Patch\n*** Update File: artifacts/research_notes.md\n"
                                      "@@\n-reviewed\n+delivered\n*** End Patch"}, "patch"),
        call("finish", {"content": "文件已整理"}, "finish")
    ])

    async def completion(self, messages, *, tools=None):
        return {"role": "assistant", "tool_calls": [next(responses)]}

    monkeypatch.setattr(Model, "completion", completion)
    events = LiveEvents(run, "file-policy")
    try:
        asyncio.run(research_turn(run, "fixture", "整理交付文件", "research", events=events))
        assert (run / "workspace/artifacts/research_notes.md").read_text() == "delivered\n"
        assert all(row["data"]["ok"] for row in events.since(0) if row["type"] == "agent.tool.end")
    finally:
        events.close()


def test_restored_worker_cannot_reopen_retired_permissions(tmp_path):
    """旧授权快照和旧工具装配均不能恢复已关闭的 record、task 与汇报别名。"""
    async def scenario():
        root = tmp_path / "runtime"

        def configure_legacy(session):
            TaskTools.register(session, storage_root=tmp_path / "tasks")
            session.loop.tools.register_callable(
                "report_progress", lambda: "should not run", description="旧汇报别名",
            )

        factory = SessionFactory(configure=configure_legacy,
                                 model_factory=lambda parent, request: FixtureModel(finished))
        parent = root_session(tmp_path)
        configure_legacy(parent)
        runtime = AgentRuntime(root, factory)
        runtime.attach(parent)
        await parent.start()
        identity = await new_worker(parent)
        saved = runtime.store.get("worker", identity)
        saved["allow_tools"] = sorted(set(saved["allow_tools"]) | DISABLED)
        with runtime.store.connection:
            runtime.store.put("worker", identity, saved)
        await runtime.aclose()
        await parent.aclose()

        parent = root_session(tmp_path)
        configure_legacy(parent)
        async with AgentRuntime(root, factory) as runtime:
            runtime.attach(parent)
            async with parent:
                await assign(parent, identity)
                result = await collect(parent, identity)
                assert result["result"]["outcome"] == "completed"
                child = runtime._sessions[identity]
                assert not DISABLED & set(child.tool_policy.allowed_names)
                assert not DISABLED & set(runtime.store.get("worker", identity)["allow_tools"])
                for name in DISABLED:
                    with pytest.raises((UnknownToolError, PermissionError)):
                        await child.loop.tools.call(name, {})
    asyncio.run(scenario())


def test_new_tools_are_not_implicitly_approved():
    """宿主新登记的工具不会因 all 或继承而自动进入角色白名单。"""
    available = COMMON | COMPUTATION | DISABLED | {"new_execution_tool"}
    assert set(role_tool_names("router", available)) == COMMON | ROUTER_SKILLS
    assert set(role_tool_names("worker", available)) == COMMON | COMPUTATION
