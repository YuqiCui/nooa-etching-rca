"""检查核心 Agent 到 Web 的持久化、交付和权限边界，不调用真实模型。"""

import asyncio
import json
from types import SimpleNamespace

import pytest
from test_capabilities import public_case

from factor_demo.autoresearch_agent_v2.agent.model import Model
from factor_demo.autoresearch_web import register_progress_tool, research_turn, restore_state
from factor_demo.live_events import LiveEvents


def call(name, arguments, identity):
    """构造模拟模型的原生工具调用。"""
    return {"id": identity, "type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments),
    }}


def configure(monkeypatch):
    """使用无凭据的模型配置。"""
    monkeypatch.setattr("factor_demo.autoresearch_web.load_provider", lambda name: SimpleNamespace(
        model="fixture", api_style="openai", api_key="fixture", base_url="https://invalid.test",
    ))


def test_turn_persists_tasks_results_and_frontend_events(tmp_path, monkeypatch):
    configure(monkeypatch)
    run = public_case(tmp_path / "case")
    responses = iter([
        [call("task_create", {"subject": "01 核查", "description": "目的、方法与验证"}, "t1"),
         call("write_file", {"path": "artifacts/01_check/report.md", "content": "研究结果"}, "w1")],
        [call("task_update", {"task_id": "T1", "status": "completed"}, "t2"),
         call("finish", {"content": "已完成，见 [报告](artifacts/01_check/report.md)"}, "f1")],
        [call("task_list", {}, "l1")],
        [call("finish", {"content": "已有研究已保留"}, "f2")],
    ])
    histories = []

    async def completion(self, messages, *, tools=None):
        histories.append(messages)
        return {"role": "assistant", "tool_calls": next(responses)}

    monkeypatch.setattr(Model, "completion", completion)
    events = LiveEvents(run, "first")
    asyncio.run(research_turn(run, "fixture", "执行研究", "research", events=events))
    saved = json.loads((run / ".autoresearch/session.json").read_text())
    assert saved["messages"][-1]["role"] == "tool"
    assert (run / "workspace/artifacts/01_check/report.md").read_text() == "研究结果"
    rows = events.since(0)
    assert any(r["type"] == "reply.commit" and "已完成" in r["data"]["markdown"] for r in rows)
    assert any(r["type"] == "agent.tool.end" and r["data"]["ok"] for r in rows)
    assert json.loads((run / "status.json").read_text())["status"] == "checkpoint"
    events.close()
    events = LiveEvents(run, "second")
    asyncio.run(research_turn(run, "fixture", "继续查看", "research", events=events))
    assert any(m.get("content", "").endswith("执行研究") for m in histories[-1])
    events.close()


def test_discussion_cannot_write_or_execute(tmp_path, monkeypatch):
    configure(monkeypatch)
    run = public_case(tmp_path / "case")
    responses = iter([
        [call("write_file", {"path": "artifacts/forbidden.md", "content": "不能保存"}, "w")],
        [call("finish", {"content": "只讨论已有结果"}, "f")],
    ])

    async def completion(self, messages, *, tools=None):
        return {"role": "assistant", "tool_calls": next(responses)}

    monkeypatch.setattr(Model, "completion", completion)
    events = LiveEvents(run, "discuss")
    asyncio.run(research_turn(run, "fixture", "解释结果", "chat", events=events))
    assert not (run / "workspace/artifacts/forbidden.md").exists()
    assert any(r["type"] == "agent.tool.end" and not r["data"]["ok"] for r in events.since(0))
    events.close()


@pytest.mark.parametrize("modes", [("research", "chat"), ("chat", "research")])
def test_web_skill_preload_matches_mode_and_survives_restore(tmp_path, monkeypatch, modes):
    """研究启动即加载流程；只读恢复保留快照，但不扩大讨论权限。"""
    configure(monkeypatch)
    run = public_case(tmp_path / "case")
    requests = 0

    async def completion(self, messages, *, tools=None):
        nonlocal requests
        requests += 1
        current_mode = modes[requests - 1]
        has_research = "research" in modes[:requests]
        names = {item["function"]["name"] for item in tools}
        loaded = [item["description"]["skill_id"] for item in self.state.skill_snapshots]
        prompt = messages[0]["content"]
        assert "# 可用技能" in prompt
        assert loaded == (["autoresearch"] if has_research else [])
        if has_research:
            assert "# 已加载技能：autoresearch" in prompt
        else:
            assert "# 已加载技能：" not in prompt
        if current_mode == "research":
            assert {"skill__list", "skill__load", "skill__read_resource"} <= names
        else:
            assert names == {
                "finish", "report_progress", "read_file", "task_list", "task_get", "record__get",
            }
        return {"role": "assistant", "tool_calls": [
            call("finish", {"content": "已核对技能与权限"}, f"finish-{requests}"),
        ]}

    monkeypatch.setattr(Model, "completion", completion)
    for index, current_mode in enumerate(modes):
        events = LiveEvents(run, f"skills-{index}")
        try:
            asyncio.run(research_turn(run, "fixture", "检查技能", current_mode, events=events))
        finally:
            events.close()
    saved = json.loads((run / ".autoresearch/session.json").read_text())
    assert [item["description"]["skill_id"] for item in saved["skill_snapshots"]] == ["autoresearch"]


def test_web_closes_skill_registry_when_setup_fails(tmp_path, monkeypatch):
    """初始化尚未创建会话就失败时，也释放批准技能目录句柄。"""
    from factor_demo.autoresearch_agent_v2.skills import SkillRegistry

    run = public_case(tmp_path / "case")
    registries = []

    def create_registry(roots):
        """保留实际目录对象，用于检查失败后的读取边界。"""
        registry = SkillRegistry(roots)
        registries.append(registry)
        return registry

    def fail_provider(name):
        """模拟模型配置初始化失败。"""
        raise ValueError("模型配置不可用")

    monkeypatch.setattr("factor_demo.autoresearch_web.SkillRegistry", create_registry)
    monkeypatch.setattr("factor_demo.autoresearch_web.load_provider", fail_provider)
    events = LiveEvents(run, "skills-failed")
    try:
        with pytest.raises(ValueError, match="模型配置不可用"):
            asyncio.run(research_turn(run, "fixture", "检查初始化", "research", events=events))
        assert len(registries) == 1
        with pytest.raises(RuntimeError, match="closed"):
            registries[0].load("autoresearch")
    finally:
        events.close()


@pytest.mark.parametrize("mode", ["research", "chat"])
def test_progress_is_published_before_finish_and_replayed(tmp_path, monkeypatch, mode):
    """进展只需正文参数，立即发布且不终止后续执行，恢复后仍可读取。"""
    configure(monkeypatch)
    run = public_case(tmp_path / "case")
    content = "已检查数据。\n\n|指标|值|\n|---|---|\n|样本|2|"
    requests = 0
    events = LiveEvents(run, "progress")

    async def completion(self, messages, *, tools=None):
        nonlocal requests
        requests += 1
        schema = next(t["function"] for t in tools if t["function"]["name"] == "report_progress")
        assert schema["parameters"]["required"] == ["content"]
        assert set(schema["parameters"]["properties"]) == {"content"}
        assert schema["parameters"]["properties"]["content"]["type"] == "string"
        if requests == 1:
            return {"role": "assistant", "tool_calls": [
                call("report_progress", {"content": content}, "progress-1"),
                call("report_progress", {"content": content}, "progress-2"),
            ]}
        assert requests == 2
        replies = [r for r in events.since(0) if r["type"] == "reply.commit"]
        assert len(replies) == 2
        assert all(r["data"]["publication"] == "progress" for r in replies)
        assert replies[0]["data"]["message_id"] != replies[1]["data"]["message_id"]
        assert all(json.loads(m["content"])["delivered"] for m in messages if m["role"] == "tool")
        return {"role": "assistant", "tool_calls": [
            call("finish", {"content": "分析完成"}, "final"),
        ]}

    monkeypatch.setattr(Model, "completion", completion)
    try:
        asyncio.run(research_turn(run, "fixture", "汇报检查进展", mode, events=events))
    finally:
        events.close()
    restored = LiveEvents(run, "progress")
    try:
        replies = [r for r in restored.since(0) if r["type"] == "reply.commit"]
        assert [r["data"]["publication"] for r in replies] == ["progress", "progress", "final"]
        assert replies[0]["data"]["markdown"] == content
        assert replies[-1]["data"]["markdown"] == "分析完成"
    finally:
        restored.close()


@pytest.mark.parametrize("content", [None, 12, {}, "", " \n"])
def test_invalid_progress_does_not_publish(tmp_path, monkeypatch, content):
    """非字符串或空白正文返回工具错误，不生成进展消息。"""
    configure(monkeypatch)
    run = public_case(tmp_path / "case")
    responses = iter([
        [call("report_progress", {"content": content}, "invalid")],
        [call("finish", {"content": "已处理参数错误"}, "final")],
    ])

    async def completion(self, messages, *, tools=None):
        return {"role": "assistant", "tool_calls": next(responses)}

    monkeypatch.setattr(Model, "completion", completion)
    events = LiveEvents(run, "invalid")
    try:
        asyncio.run(research_turn(run, "fixture", "检查参数", "chat", events=events))
        rows = events.since(0)
        assert not any(r["type"] == "reply.commit" and r["data"]["publication"] == "progress"
                       for r in rows)
        assert any(r["type"] == "agent.tool.end" and r["data"]["name"] == "report_progress"
                   and not r["data"]["ok"] for r in rows)
    finally:
        events.close()


def test_child_progress_keeps_agent_identity(tmp_path, monkeypatch):
    """获准子会话发布自己的进展，结束结果仍只投递父会话。"""
    configure(monkeypatch)
    run = public_case(tmp_path / "case")
    root_calls = 0
    child_calls = 0

    async def completion(self, messages, *, tools=None):
        nonlocal root_calls, child_calls
        if self.state.parent_agent_id is not None:
            child_calls += 1
            assert {t["function"]["name"] for t in tools} == {"report_progress", "finish"}
            if child_calls == 1:
                return {"role": "assistant", "tool_calls": [
                    call("report_progress", {"content": "子任务正在核查"}, "progress"),
                ]}
            return {"role": "assistant", "tool_calls": [
                call("finish", {"content": "子任务结果"}, "child-finish"),
            ]}
        root_calls += 1
        assert root_calls < 8
        if root_calls == 1:
            return {"role": "assistant", "tool_calls": [
                call("agent__spawn", {
                    "task": "独立核查", "description": "核查已有发现",
                    "allow_tools": ["report_progress", "finish"],
                }, "spawn"),
            ]}
        if not any(str(m.get("content", "")).startswith("子会话结果") for m in messages):
            return {"role": "assistant", "tool_calls": [call("agent__wait", {}, "wait")]}
        return {"role": "assistant", "tool_calls": [
            call("finish", {"content": "已汇总子任务结果"}, "root-finish"),
        ]}

    monkeypatch.setattr(Model, "completion", completion)
    events = LiveEvents(run, "child-progress")
    try:
        asyncio.run(research_turn(run, "fixture", "委派并汇报", "research", events=events))
        replies = [r for r in events.since(0) if r["type"] == "reply.commit"]
        assert len(replies) == 2
        assert replies[0]["agent_id"] != "main"
        assert replies[0]["parent_agent_id"] == "main"
        assert replies[0]["data"]["publication"] == "progress"
        assert replies[-1]["agent_id"] == "main"
        assert replies[-1]["data"]["publication"] == "final"
    finally:
        events.close()


def test_progress_respects_policy_and_reuses_call_identity(tmp_path):
    """授权控制真实调用，重复执行同一调用时使用相同消息标识。"""
    from factor_demo.autoresearch_agent_v2.agent import Session
    from factor_demo.autoresearch_agent_v2.config import ModelConfig

    run = public_case(tmp_path / "case")
    events = LiveEvents(run, "policy")

    async def scenario():
        state = restore_state(run / "missing", run)
        model = Model(ModelConfig(model="openai/fixture", api_key="fixture"))
        blocked = Session(model, state, allow_tools=["finish"])
        register_progress_tool(blocked, events)
        async with blocked:
            assert all(t["function"]["name"] != "report_progress" for t in blocked.schemas())
            assert "使用 report_progress(content) 向用户汇报阶段性进展" not in (
                blocked.loop.build_system_prompt()
            )
            with pytest.raises(PermissionError):
                await blocked.call("report_progress", {"content": "不应发布"})
        assert events.since(0) == []
        enabled = Session(model, state)
        register_progress_tool(enabled, events)
        async with enabled:
            first = await enabled.loop.tools.call(
                "report_progress", {"content": "初次进展"}, call_id="same-call",
            )
            repeated = await enabled.loop.tools.call(
                "report_progress", {"content": "更新进展"}, call_id="same-call",
            )
            assert first["message_id"] == repeated["message_id"]
            events.close()
            with pytest.raises(ValueError, match="通道已经关闭"):
                await enabled.call("report_progress", {"content": "不应假报投递成功"})

    try:
        asyncio.run(scenario())
    finally:
        events.close()


def test_cancellation_preserves_published_progress(tmp_path, monkeypatch):
    """进展发布后取消研究，已提交消息仍保留且不补发最终结论。"""
    configure(monkeypatch)
    run = public_case(tmp_path / "case")
    calls = 0

    async def completion(self, messages, *, tools=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"role": "assistant", "tool_calls": [
                call("report_progress", {"content": "已有检查结果"}, "progress"),
            ]}
        raise asyncio.CancelledError

    monkeypatch.setattr(Model, "completion", completion)
    events = LiveEvents(run, "cancel-progress")
    try:
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(research_turn(run, "fixture", "开始检查", "research", events=events))
        replies = [r for r in events.since(0) if r["type"] == "reply.commit"]
        assert len(replies) == 1
        assert replies[0]["data"]["publication"] == "progress"
        assert replies[0]["data"]["markdown"] == "已有检查结果"
        assert json.loads((run / "status.json").read_text())["status"] == "cancelled"
    finally:
        events.close()


def test_stop_saves_cancelled_checkpoint(tmp_path, monkeypatch):
    configure(monkeypatch)
    run = public_case(tmp_path / "case")
    (run / "requests").mkdir()
    (run / "requests/stop.cancel").touch()

    async def completion(self, messages, *, tools=None):
        await asyncio.Event().wait()

    monkeypatch.setattr(Model, "completion", completion)
    events = LiveEvents(run, "stop")
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(research_turn(run, "fixture", "研究", "research", events=events))
    assert json.loads((run / "status.json").read_text())["status"] == "cancelled"
    assert (run / ".autoresearch/session.json").exists()
    events.close()


def test_restore_marks_unfinished_call_without_repeating_it(tmp_path):
    state = restore_state(tmp_path / "missing", tmp_path)
    state.messages.append({"role": "assistant", "tool_calls": [call("write_file", {}, "w")]})
    path = tmp_path / "state.json"
    path.write_text(state.model_dump_json())
    restored = restore_state(path, tmp_path)
    assert restored.messages[-1]["tool_call_id"] == "w"
    assert "未确认" in restored.messages[-1]["content"]


def test_web_routes_parallel_child_results_to_parent_and_keeps_json_owned(tmp_path, monkeypatch):
    """真实 Web 装配支持并行回传，子 finish 不直接发布用户回复。"""
    from factor_demo.autoresearch_agent_v2.tools.builtin_tools.records.store import RecordStore

    configure(monkeypatch)
    run = public_case(tmp_path / "case")
    entered = set()
    both_entered = asyncio.Event()
    parent_calls = 0

    async def completion(self, messages, *, tools=None):
        nonlocal parent_calls
        if self.state.parent_agent_id is not None:
            assert {item["function"]["name"] for item in tools} == {"finish"}
            assert "# 已加载技能：research-executor" in messages[0]["content"]
            assert "# 已加载技能：autoresearch" not in messages[0]["content"]
            entered.add(self.state.agent_id)
            if len(entered) == 2:
                both_entered.set()
            await asyncio.wait_for(both_entered.wait(), 3)
            return {"role": "assistant", "tool_calls": [
                call("finish", {"content": "子任务已完成"}, "child-finish"),
            ]}
        parent_calls += 1
        assert parent_calls < 8
        if parent_calls == 1:
            assert "# 已加载技能：autoresearch" in messages[0]["content"]
            return {"role": "assistant", "tool_calls": [
                call("agent__spawn", {
                    "task": "检查甲", "description": "核查甲项", "allow_tools": ["finish"],
                    "skills": ["research-executor"],
                }, "a"),
                call("agent__spawn", {
                    "task": "检查乙", "description": "核查乙项", "allow_tools": ["finish"],
                    "skills": ["research-executor"],
                }, "b"),
                call("finish", {"content": "尚未收到结果"}, "early"),
            ]}
        results = [m for m in messages if str(m.get("content", "")).startswith("子会话结果")]
        if len(results) < 2:
            return {"role": "assistant", "tool_calls": [call("agent__wait", {}, "wait")]}
        return {"role": "assistant", "tool_calls": [
            call("record__put", {
                "document_id": "research.json", "content": {"received": len(results)},
                "expected_revision": 0,
            }, "record"),
            call("finish", {"content": "两项结果已收取"}, "root-finish"),
        ]}

    monkeypatch.setattr(Model, "completion", completion)
    events = LiveEvents(run, "parallel")
    try:
        asyncio.run(research_turn(run, "fixture", "并行处理两项任务", "research", events=events))
        saved = json.loads((run / ".autoresearch/session.json").read_text())
        assert len(saved["consumed_event_ids"]) == 2
        rows = events.since(0)
        child_tools = [r for r in rows if r["type"] == "agent.tool.end" and r["agent_id"] != "main"]
        assert len(child_tools) == 2
        replies = [r for r in rows if r["type"] == "reply.commit"]
        assert len(replies) == 1
        assert "两项结果已收取" in replies[0]["data"]["markdown"]
        store = RecordStore(run / ".autoresearch/records")
        try:
            record = store.get(run.name, "main", "research.json")
            assert record["document"]["content"] == {"received": 2}
        finally:
            store.close()
    finally:
        events.close()


def test_discussion_does_not_resume_interrupted_child_tasks(tmp_path, monkeypatch):
    """只读讨论不会因运行目录中存在未完成委派而自动启动研究。"""
    from factor_demo.autoresearch_agent_v2.agent import Session, SessionState
    from factor_demo.autoresearch_agent_v2.config import ModelConfig
    from factor_demo.autoresearch_agent_v2.database import RuntimeStore
    from factor_demo.autoresearch_agent_v2.runtime import AgentRuntime

    configure(monkeypatch)
    run = public_case(tmp_path / "case")
    storage = run / ".autoresearch/runtime"

    async def prepare():
        state = SessionState(
            session_id="previous", agent_id="main", task_id=run.name, workdir=run / "workspace",
        )
        parent = Session(Model(ModelConfig(model="openai/fixture", api_key="fixture")), state)
        runtime = AgentRuntime(storage)
        runtime.attach(parent)
        try:
            async with parent:
                queued = await runtime.spawn(parent, "不能在讨论模式恢复", allow_tools=["finish"])
                await runtime.aclose()
                return queued["agent_id"]
        finally:
            await runtime.aclose()

    child_id = asyncio.run(prepare())

    async def completion(self, messages, *, tools=None):
        assert self.state.agent_id == "main"
        assert all(not tool["function"]["name"].startswith("agent__") for tool in tools)
        return {"role": "assistant", "tool_calls": [
            call("finish", {"content": "仅解释已有材料"}, "discussion"),
        ]}

    monkeypatch.setattr(Model, "completion", completion)
    events = LiveEvents(run, "discussion")
    try:
        asyncio.run(research_turn(run, "fixture", "解释已有结果", "chat", events=events))
        store = RuntimeStore(storage)
        try:
            assert store.agent(child_id)["status"] == "interrupted"
        finally:
            store.close()
    finally:
        events.close()


@pytest.mark.parametrize("engine", ["autoresearch_v2", "legacy"])
def test_web_worker_routes_by_saved_session_engine(tmp_path, monkeypatch, engine):
    from factor_demo import web_worker

    run = public_case(tmp_path / "case")
    (run / "requests").mkdir()
    (run / "ui.json").write_text(json.dumps({"engine": engine}))
    (run / "requests/request.json").write_text(json.dumps({
        "message": "研究", "mode": "research", "provider": "fixture",
    }))
    selected = []

    async def old(*args, **kwargs):
        selected.append("legacy")

    async def core(*args, **kwargs):
        selected.append("autoresearch_v2")

    monkeypatch.setattr(web_worker, "research_turn", old)
    monkeypatch.setattr("factor_demo.autoresearch_web.research_turn", core)
    asyncio.run(web_worker.main(SimpleNamespace(run=run, task="request")))
    assert selected == [engine]
