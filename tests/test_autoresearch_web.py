"""检查核心 Agent 到 Web 的持久化、交付和权限边界，不调用真实模型。"""

import asyncio
import json
from types import SimpleNamespace

import pytest
from test_capabilities import public_case

from factor_demo.autoresearch_agent_v2.agent.model import Model
from factor_demo.autoresearch_web import research_turn, restore_state
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
