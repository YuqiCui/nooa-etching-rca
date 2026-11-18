"""检查核心 Agent 到 Web 的持久化、交付和权限边界，不调用真实模型。"""

import asyncio
import json
from types import SimpleNamespace

import pytest
from test_capabilities import public_case

from factor_demo.autoresearch_agent_v2.agent.model import Model
from factor_demo.autoresearch_web import register_progress_tool, research_turn, restore_state
from factor_demo.live_events import LiveEvents
from factor_demo.runtime import RuntimeSettings


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


@pytest.mark.parametrize("timeout", [None, 0.02])
def test_task_timeout_uses_config_or_explicit_override_and_saves_checkpoint(
    tmp_path, monkeypatch, timeout,
):
    """总时限取消卡住的模型，并持久化超时状态和已有对话。"""
    configure(monkeypatch)
    settings = RuntimeSettings(task_timeout=1, router_time_budget=1800, router_token_budget=1_000)
    monkeypatch.setattr(RuntimeSettings, "load", classmethod(lambda cls, path: settings))
    run = public_case(tmp_path / "case")
    calls = cancelled = 0

    async def completion(self, messages, *, tools=None):
        nonlocal calls, cancelled
        calls += 1
        assert self.config.timeout_s == 60
        assert self.config.num_retries == 10
        assert self.config.retry_initial_delay_s == 1
        assert self.config.retry_max_delay_s == 300
        assert "Router 时间预算 30 分钟" in messages[0]["content"]
        assert "Router token 预算：1000" in messages[0]["content"]
        try:
            await asyncio.Event().wait()
        finally:
            cancelled += 1

    monkeypatch.setattr(Model, "completion", completion)
    events = LiveEvents(run, "timeout")
    try:
        options = {} if timeout is None else {"timeout": timeout}
        with pytest.raises(TimeoutError):
            asyncio.run(research_turn(
                run, "fixture", "检查超时保存", "chat", events=events, **options,
            ))
    finally:
        events.close()
    assert calls == cancelled == 1
    assert json.loads((run / "status.json").read_text())["status"] == "timeout"
    checkpoint = json.loads((run / ".autoresearch/session.json").read_text())
    assert any("检查超时保存" in str(message) for message in checkpoint["messages"])


def test_runtime_config_compacts_reused_web_worker_with_summary_tools(tmp_path, monkeypatch):
    """Web 配置传入子会话，压缩请求保留工具且使用独立选择策略。"""
    from test_agent_runtime import delivery

    from factor_demo.autoresearch_agent_v2.agent.history import COMPACT_PROMPT, SUMMARY_PREFIX

    configure(monkeypatch)
    config_path = tmp_path / "runtime.json"
    config_path.write_text(json.dumps({"compact_on_worker_reassignment": True}))
    load = RuntimeSettings.load
    monkeypatch.setattr(RuntimeSettings, "load", classmethod(lambda cls, path: load(config_path)))
    run = public_case(tmp_path / "case")
    root_calls = 0
    worker_id = None
    summaries = []
    worker_requests = []

    async def completion(self, messages, *, tools=None, tool_choice=None):
        nonlocal root_calls, worker_id
        if self.state.parent_agent_id is not None:
            if messages[-1].get("content") == COMPACT_PROMPT:
                summaries.append(messages)
                assert tools and tool_choice == "auto"
                return {"role": "assistant", "content": "First assignment completed successfully."}
            worker_requests.append(messages)
            return {"role": "assistant", "tool_calls": [call("finish", {"content": delivery()}, "finish")]}
        root_calls += 1
        assert root_calls <= 8
        if root_calls in (5, 8):
            assert json.loads(messages[-1]["content"])["result"]["outcome"] == "completed"
        if root_calls == 1:
            calls = [call("StartWorker", {"worker_type": "general"}, "start")]
        elif root_calls in (2, 5):
            if root_calls == 2:
                worker_id = json.loads(messages[-1]["content"])["worker_id"]
            calls = [call("AssignTask", {
                "worker_id": worker_id, "task": f"assignment-{root_calls}",
                "deliver_list": ["result"],
            }, f"assign-{root_calls}")]
        elif root_calls in (3, 6):
            calls = [call("WaitWorkers", {}, f"wait-{root_calls}")]
        elif root_calls in (4, 7):
            calls = [call("CollectWorkerResult", {"worker_id": worker_id}, f"collect-{root_calls}")]
        else:
            return {"role": "assistant", "tool_calls": [call("finish", {"content": "done"}, "finish")]}
        return {"role": "assistant", "tool_calls": calls}

    monkeypatch.setattr(Model, "completion", completion)
    events = LiveEvents(run, "reassignment-compaction")
    try:
        asyncio.run(asyncio.wait_for(research_turn(
            run, "fixture", "reuse worker", "research", events=events,
        ), 5))
        assert len(summaries) == 1 and len(worker_requests) == 2
        assert worker_requests[1][1]["content"].startswith(SUMMARY_PREFIX)
        assert "assignment-5" in worker_requests[1][-1]["content"]
        path = run / ".autoresearch/children" / worker_id / "session.json"
        assert json.loads(path.read_text())["compaction"]["summary"] == (
            "First assignment completed successfully."
        )
    finally:
        events.close()


def test_turn_persists_json_files_results_and_frontend_events(tmp_path, monkeypatch):
    """研究 JSON 通过文件工具更新，恢复后的讨论可读取真实文件。"""
    configure(monkeypatch)
    run = public_case(tmp_path / "case")
    responses = iter([
        [call("write_file", {"path": "research.json", "content": json.dumps({"status": "running"})}, "t1"),
         call("write_file", {"path": "artifacts/01_check/report.md", "content": "研究结果"}, "w1")],
        [call("write_file", {"path": "research.json", "content": json.dumps({"status": "completed"})}, "t2")],
        [call("finish", {"content": "已完成，见 [报告](artifacts/01_check/report.md)"}, "finish")],
        [call("read_file", {"path": "research.json"}, "l1")],
        [call("finish", {"content": "已有研究已保留"}, "finish")]
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
    assert json.loads((run / "workspace/research.json").read_text()) == {"status": "completed"}
    assert not (run / ".autoresearch/records").exists()
    rows = events.since(0)
    assert any(r["type"] == "reply.commit" and "已完成" in r["data"]["markdown"] for r in rows)
    assert any(r["type"] == "agent.tool.end" and r["data"]["ok"] for r in rows)
    assert json.loads((run / "status.json").read_text())["status"] == "checkpoint"
    events.close()
    events = LiveEvents(run, "second")
    asyncio.run(research_turn(run, "fixture", "继续查看", "chat", events=events))
    assert any(m.get("content", "").endswith("执行研究") for m in histories[-1])
    assert '"status": "completed"' in histories[-1][-1]["content"]
    events.close()


def test_discussion_cannot_write_or_execute(tmp_path, monkeypatch):
    configure(monkeypatch)
    run = public_case(tmp_path / "case")
    responses = iter([
        [call("write_file", {"path": "artifacts/forbidden.md", "content": "不能保存"}, "w")],
        [call("finish", {"content": "只讨论已有结果"}, "finish")]
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
def test_web_discovers_router_skill_without_preloading_and_drops_retired_snapshots(
    tmp_path, monkeypatch, modes,
):
    """研究模式只展示技能元数据，讨论模式不开放技能，旧流程快照不会恢复。"""
    configure(monkeypatch)
    run = public_case(tmp_path / "case")
    requests = 0

    async def completion(self, messages, *, tools=None):
        nonlocal requests
        requests += 1
        names = {item["function"]["name"] for item in tools}
        assert not self.state.skill_snapshots
        assert "# 已加载技能" not in messages[0]["content"]
        assert "# 任务分发与管理" not in messages[0]["content"]
        if modes[requests - 1] == "chat":
            assert names == {"finish", "report", "read_file"}
            assert "# 可用技能" not in messages[0]["content"]
        else:
            assert {"skill__list", "skill__load", "skill__read_resource"} <= names
            assert "feature-research" in messages[0]["content"]
        return {"role": "assistant", "tool_calls": [call("finish", {"content": "已核对权限"}, "finish")]}

    monkeypatch.setattr(Model, "completion", completion)
    for index, current_mode in enumerate(modes):
        events = LiveEvents(run, f"skills-{index}")
        try:
            asyncio.run(research_turn(run, "fixture", "检查技能", current_mode, events=events))
        finally:
            events.close()
        saved_path = run / ".autoresearch/session.json"
        saved = json.loads(saved_path.read_text())
        assert saved["skill_snapshots"] == []
        # 模拟历史部署留下的技能快照；当前入口必须在装配前清除。
        saved["skill_snapshots"] = [{"retired": "旧流程不能重新加载"}]
        saved_path.write_text(json.dumps(saved))


def test_web_router_loads_skill_on_demand_and_restores_approved_snapshot(tmp_path, monkeypatch):
    """系统提示词始终只有名称和描述，正文通过加载工具返回并随历史恢复。"""
    configure(monkeypatch)
    run = public_case(tmp_path / "case")
    requests = 0

    async def completion(self, messages, *, tools=None):
        nonlocal requests
        requests += 1
        prompt = messages[0]["content"]
        assert "# 任务分发与管理" not in prompt
        catalog = json.loads(prompt.split("# 可用技能\n", 1)[1].split("\n\n", 1)[0])
        assert len(catalog) == 1
        assert set(catalog[0]) == {"name", "description"}
        assert catalog[0]["name"] == "feature-research"
        if requests == 1:
            assert "feature-research" in prompt
            assert "# 任务分发与管理" not in prompt
            return {"role": "assistant", "tool_calls": [
                call("skill__load", {"skill_id": "feature-research"}, "load"),
            ]}
        skill_results = [
            json.loads(message["content"]) for message in messages
            if message["role"] == "tool" and message.get("tool_call_id") == "load"
        ]
        assert len(skill_results) == 1
        assert "# 任务分发与管理" in skill_results[0]["content"]
        assert len(self.state.skill_snapshots) == 1
        return {"role": "assistant", "tool_calls": [
            call("finish", {"content": "研究流程已加载"}, f"finish-{requests}"),
        ]}

    monkeypatch.setattr(Model, "completion", completion)
    for index in range(2):
        events = LiveEvents(run, f"load-skill-{index}")
        try:
            asyncio.run(research_turn(run, "fixture", "继续特征分析", "research", events=events))
        finally:
            events.close()
    assert requests == 3
    saved = json.loads((run / ".autoresearch/session.json").read_text())
    assert saved["skill_snapshots"][0]["description"]["skill_id"] == "feature-research"
    assert sum("# 任务分发与管理" in str(message) for message in saved["messages"]) == 1


def test_web_setup_failure_closes_skill_registry(tmp_path, monkeypatch):
    """模型配置失败时也释放已批准技能目录的文件句柄。"""
    from factor_demo.autoresearch_agent_v2.skills import SkillRegistry

    configure(monkeypatch)
    run = public_case(tmp_path / "case")
    registries = []

    def track_registry(*args, **kwargs):
        """保存实际注册器以检查异常路径的资源清理。"""
        registry = SkillRegistry(*args, **kwargs)
        registries.append(registry)
        return registry

    def fail_provider(name):
        """模拟模型配置初始化失败。"""
        raise ValueError("模型配置不可用")

    monkeypatch.setattr(
        "factor_demo.autoresearch_web.SkillRegistry", track_registry,
    )
    monkeypatch.setattr("factor_demo.autoresearch_web.load_provider", fail_provider)
    events = LiveEvents(run, "skills-failed")
    try:
        with pytest.raises(ValueError, match="模型配置不可用"):
            asyncio.run(research_turn(run, "fixture", "检查初始化", "research", events=events))
    finally:
        events.close()
    assert len(registries) == 1
    with pytest.raises(RuntimeError, match="closed"):
        registries[0].load("feature-research")


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
        schema = next(t["function"] for t in tools if t["function"]["name"] == "report")
        assert schema["parameters"]["required"] == ["content"]
        assert set(schema["parameters"]["properties"]) == {"content"}
        assert schema["parameters"]["properties"]["content"]["type"] == "string"
        if requests == 1:
            return {"role": "assistant", "tool_calls": [
                call("report", {"content": content}, "progress-1"),
                call("report", {"content": content}, "progress-2"),
            ]}
        assert requests == 2
        replies = [r for r in events.since(0) if r["type"] == "reply.commit"]
        assert len(replies) == 2
        assert all(r["data"]["publication"] == "progress" for r in replies)
        assert replies[0]["data"]["message_id"] != replies[1]["data"]["message_id"]
        assert all(json.loads(m["content"])["delivered"] for m in messages if m["role"] == "tool")
        return {"role": "assistant", "tool_calls": [call("finish", {"content": "分析完成"}, "finish")]}

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
        [call("report", {"content": content}, "invalid")],
        [call("finish", {"content": "已处理参数错误"}, "finish")]
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
        assert any(r["type"] == "agent.tool.end" and r["data"]["name"] == "report"
                   and not r["data"]["ok"] for r in rows)
    finally:
        events.close()


def test_child_progress_keeps_agent_identity(tmp_path, monkeypatch):
    """Worker 的进展保留身份，结构化结果经收集后才由 Router 交付。"""
    from test_agent_runtime import delivery

    configure(monkeypatch)
    run = public_case(tmp_path / "case")
    root_calls = 0
    child_calls = 0
    worker_id = None

    async def completion(self, messages, *, tools=None):
        nonlocal root_calls, child_calls, worker_id
        names = {t["function"]["name"] for t in tools}
        assert not any(name.startswith("agent__") for name in names)
        if self.state.parent_agent_id is not None:
            child_calls += 1
            if child_calls == 1:
                return {"role": "assistant", "tool_calls": [
                    call("report", {"content": "子任务正在核查"}, "progress"),
                ]}
            return {"role": "assistant", "tool_calls": [call("finish", {"content": delivery("子任务结果")}, "finish")]}
        root_calls += 1
        assert root_calls < 8
        if root_calls == 1:
            calls = [call("StartWorker", {"worker_type": "general"}, "start")]
        elif root_calls == 2:
            worker_id = json.loads(messages[-1]["content"])["worker_id"]
            calls = [call("AssignTask", {"worker_id": worker_id, "task": "独立核查",
                                         "deliver_list": ["核查结果"]}, "assign")]
        elif root_calls == 3:
            calls = [call("WaitWorkers", {}, "wait")]
        elif root_calls == 4:
            calls = [call("CollectWorkerResult", {"worker_id": worker_id}, "collect")]
        else:
            return {"role": "assistant", "tool_calls": [call("finish", {"content": "已汇总子任务结果"}, "finish")]}
        return {"role": "assistant", "tool_calls": calls}

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
            assert all(t["function"]["name"] != "report" for t in blocked.schemas())
            assert "使用 report(content) 向用户汇报阶段性进展" not in (
                blocked.loop.build_system_prompt()
            )
            with pytest.raises(PermissionError):
                await blocked.call("report", {"content": "不应发布"})
        assert events.since(0) == []
        enabled = Session(model, state)
        register_progress_tool(enabled, events)
        async with enabled:
            first = await enabled.loop.tools.call(
                "report", {"content": "初次进展"}, call_id="same-call",
            )
            repeated = await enabled.loop.tools.call(
                "report", {"content": "更新进展"}, call_id="same-call",
            )
            assert first["message_id"] == repeated["message_id"]
            events.close()
            with pytest.raises(ValueError, match="通道已经关闭"):
                await enabled.call("report", {"content": "不应假报投递成功"})

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
                call("report", {"content": "已有检查结果"}, "progress"),
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


def test_web_routes_parallel_child_results_to_parent_and_saves_json_file(tmp_path, monkeypatch):
    """Router 创建多个 Worker，显式收集后保存全局 JSON 文件，工具事件不重复。"""
    from test_agent_runtime import delivery

    configure(monkeypatch)
    run = public_case(tmp_path / "case")
    entered = set()
    both_entered = asyncio.Event()
    parent_calls = 0
    worker_ids = []

    async def completion(self, messages, *, tools=None):
        nonlocal parent_calls, worker_ids
        if self.state.parent_agent_id is not None:
            assert "You are a Worker." in messages[0]["content"]
            assert "You are a Router." not in messages[0]["content"]
            entered.add(self.state.agent_id)
            if len(entered) == 2:
                both_entered.set()
            await asyncio.wait_for(both_entered.wait(), 3)
            return {"role": "assistant", "tool_calls": [call("finish", {"content": delivery("子任务已完成")}, "finish")]}
        parent_calls += 1
        assert parent_calls < 15
        assert "You are a Router." in messages[0]["content"]
        if parent_calls == 1:
            return {"role": "assistant", "tool_calls": [
                call("StartWorker", {"worker_type": "general"}, "a"),
                call("StartWorker", {"worker_type": "general"}, "b"),
            ]}
        if parent_calls == 2:
            worker_ids = [json.loads(m["content"])["worker_id"] for m in messages[-2:]]
            return {"role": "assistant", "tool_calls": [
                *[call("AssignTask", {"worker_id": worker, "task": "检查输入与证据",
                                       "deliver_list": ["检查结果"]}, f"assign-{worker}")
                  for worker in worker_ids],
                call("finish", {"content": "尚未收到结果"}, "early-finish"),
            ]}
        collected, ready = set(), set()
        for message in messages:
            content = message.get("content") or ""
            if message["role"] == "tool":
                try:
                    result = json.loads(content)
                except (ValueError, TypeError):
                    continue
                if isinstance(result, dict) and result.get("ok"):
                    if "result" in result:
                        collected.add(result["worker_id"])
                    ready.update(item["worker_id"] for item in result.get("ready_workers", []))
            if content.startswith("Worker runtime notification"):
                ready.add(json.loads(content.split("\n", 1)[1])["worker_id"])
        if len(collected) == 2:
            return {"role": "assistant", "tool_calls": [
                call("write_file", {"path": "research.json",
                                    "content": json.dumps({"received": 2})}, "save-json"),
                call("finish", {"content": "两项结果已收取"}, "finish"),
            ]}
        calls = [call("CollectWorkerResult", {"worker_id": worker}, f"collect-{worker}")
                 for worker in sorted(ready - collected)]
        return {"role": "assistant", "tool_calls": calls or [
            call("WaitWorkers", {}, f"wait-{parent_calls}"),
        ]}

    monkeypatch.setattr(Model, "completion", completion)
    events = LiveEvents(run, "parallel")
    try:
        asyncio.run(research_turn(run, "fixture", "并行处理两项任务", "research", events=events))
        rows = events.since(0)
        child_tools = [r for r in rows if r["type"] == "agent.tool.end" and r["agent_id"] != "main"]
        assert len(child_tools) == 2
        assert len([r for r in rows if r["type"] == "agent.tool.end"
                    and r["data"]["name"] == "CollectWorkerResult"]) == 2
        replies = [r for r in rows if r["type"] == "reply.commit"]
        assert len(replies) == 1
        assert "两项结果已收取" in replies[0]["data"]["markdown"]
        assert json.loads((run / "workspace/research.json").read_text()) == {"received": 2}
    finally:
        events.close()


def test_discussion_does_not_resume_interrupted_child_tasks(tmp_path, monkeypatch):
    """只读讨论不会因运行目录中存在未完成委派而自动启动研究。"""
    from factor_demo.autoresearch_agent_v2.agent import Session, SessionState
    from factor_demo.autoresearch_agent_v2.config import ModelConfig
    from factor_demo.autoresearch_agent_v2.database import WorkerStore
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
                worker = await parent.call("StartWorker", {"worker_type": "general"})
                queued = await parent.call("AssignTask", {
                    "worker_id": worker["worker_id"], "task": "不能在讨论模式恢复",
                    "deliver_list": ["检查结果"],
                })
                await runtime.aclose()
                return queued["worker_id"]
        finally:
            await runtime.aclose()

    child_id = asyncio.run(prepare())

    async def completion(self, messages, *, tools=None):
        assert self.state.agent_id == "main"
        assert all(not tool["function"]["name"].startswith("agent__") for tool in tools)
        return {"role": "assistant", "tool_calls": [call("finish", {"content": "仅解释已有材料"}, "finish")]}

    monkeypatch.setattr(Model, "completion", completion)
    events = LiveEvents(run, "discussion")
    try:
        asyncio.run(research_turn(run, "fixture", "解释已有结果", "chat", events=events))
        store = WorkerStore(storage)
        try:
            assert store.get("worker", child_id)["status"] == "busy"
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


def test_reused_worker_progress_keeps_assignment_identity(tmp_path, monkeypatch):
    """复用 Worker 不重复挂事件，重复工具 ID 不覆盖上一项任务的进展。"""
    from test_agent_runtime import delivery

    configure(monkeypatch)
    load = RuntimeSettings.load
    monkeypatch.setattr(RuntimeSettings, "load", classmethod(
        lambda cls, path: load(path).model_copy(update={"compact_on_worker_reassignment": False})
    ))
    run = public_case(tmp_path / "case")
    root_calls = 0
    child_calls = {}
    identity = None

    async def completion(self, messages, *, tools=None):
        nonlocal root_calls, identity
        assignment = self.state.assignment_id
        if self.state.parent_agent_id is not None:
            child_calls[assignment] = child_calls.get(assignment, 0) + 1
            if child_calls[assignment] == 1:
                calls = [call("report", {"content": f"检查中：{assignment}"}, "progress")]
            else:
                return {"role": "assistant", "tool_calls": [call("finish", {"content": delivery()}, "finish")]}
            return {"role": "assistant", "tool_calls": calls}
        root_calls += 1
        assert root_calls <= 8
        if root_calls == 1:
            calls = [call("StartWorker", {"worker_type": "general"}, "start")]
        elif root_calls in (2, 5):
            if root_calls == 2:
                identity = json.loads(messages[-1]["content"])["worker_id"]
            calls = [call("AssignTask", {"worker_id": identity, "task": f"检查 {root_calls}",
                                         "deliver_list": ["结果"]}, f"assign-{root_calls}")]
        elif root_calls in (3, 6):
            calls = [call("WaitWorkers", {}, f"wait-{root_calls}")]
        elif root_calls in (4, 7):
            calls = [call("CollectWorkerResult", {"worker_id": identity}, f"collect-{root_calls}")]
        else:
            return {"role": "assistant", "tool_calls": [call("finish", {"content": "两次检查已收集"}, "finish")]}
        return {"role": "assistant", "tool_calls": calls}

    monkeypatch.setattr(Model, "completion", completion)
    events = LiveEvents(run, "reuse")
    try:
        asyncio.run(research_turn(run, "fixture", "检查后补充", "research", events=events))
        rows = events.since(0)
        progress = [row for row in rows if row["type"] == "reply.commit"
                    and row["data"]["publication"] == "progress"]
        assert len(progress) == 2
        assert progress[0]["data"]["message_id"] != progress[1]["data"]["message_id"]
        assert len([row for row in rows if row["type"] == "agent.tool.end"
                    and row["data"]["name"] == "report"]) == 2
    finally:
        events.close()


def test_user_guidance_interrupts_web_worker_wait(tmp_path, monkeypatch):
    """网页人工引导中断入口等待，既有 Worker 继续运行并显式交付。"""
    from test_agent_runtime import delivery

    configure(monkeypatch)
    run = public_case(tmp_path / "case")
    entered = asyncio.Event()
    release = asyncio.Event()
    root_calls = 0
    identity = None

    async def completion(self, messages, *, tools=None):
        nonlocal root_calls, identity
        if self.state.parent_agent_id is not None:
            entered.set()
            await release.wait()
            return {"role": "assistant", "tool_calls": [call("finish", {"content": delivery()}, "finish")]}
        root_calls += 1
        assert root_calls <= 6
        if root_calls == 1:
            calls = [call("StartWorker", {"worker_type": "general"}, "start")]
        elif root_calls == 2:
            identity = json.loads(messages[-1]["content"])["worker_id"]
            calls = [call("AssignTask", {"worker_id": identity, "task": "继续检查",
                                         "deliver_list": ["结果"]}, "assign")]
        elif root_calls == 3:
            calls = [call("WaitWorkers", {}, "wait")]
        elif root_calls == 4:
            assert any(m.get("content") == "调整汇报重点" for m in messages)
            assert any(m["role"] == "tool" and '"interrupted"' in m.get("content", "")
                       for m in messages)
            release.set()
            calls = [call("WaitWorkers", {}, "wait-again")]
        elif root_calls == 5:
            calls = [call("CollectWorkerResult", {"worker_id": identity}, "collect")]
        else:
            return {"role": "assistant", "tool_calls": [call("finish", {"content": "已按引导汇报"}, "finish")]}
        return {"role": "assistant", "tool_calls": calls}

    monkeypatch.setattr(Model, "completion", completion)
    events = LiveEvents(run, "guidance")

    async def scenario():
        async def guide():
            await entered.wait()
            events.add_guidance("调整汇报重点", source="user")
        guide_task = asyncio.create_task(guide())
        try:
            await asyncio.wait_for(research_turn(
                run, "fixture", "检查数据", "research", events=events), 5)
        finally:
            guide_task.cancel()
            await asyncio.gather(guide_task, return_exceptions=True)

    try:
        asyncio.run(scenario())
        assert len([row for row in events.since(0) if row["type"] == "steering.applied"]) == 1
    finally:
        events.close()
