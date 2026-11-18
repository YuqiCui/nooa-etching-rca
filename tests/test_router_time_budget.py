"""通过可控时钟验证 Router 预算消息及工具批次边界。"""

import asyncio
from copy import deepcopy
from types import SimpleNamespace

import litellm
import pytest

from factor_demo.autoresearch_agent_v2.agent import budget
from factor_demo.autoresearch_agent_v2.agent.history import COMPACT_PROMPT, SUMMARY_PREFIX
from factor_demo.autoresearch_agent_v2.agent.loop import AgentLoop
from factor_demo.autoresearch_agent_v2.agent.model import Model
from factor_demo.autoresearch_agent_v2.agent.typing import SessionState
from factor_demo.autoresearch_agent_v2.config import HistoryConfig, ModelConfig, TimeBudgetConfig
from factor_demo.runtime import RuntimeSettings


def test_budget_thresholds_and_overrun(monkeypatch):
    """逐阈值投递一次，超时不显示负数；统计不依赖墙钟调整。"""
    clock = [100.0]
    monkeypatch.setattr(budget.time, "monotonic", lambda: clock[0])
    tracker = budget.TimeBudget(2400)
    initial = tracker.initial_prompt()
    assert "40 分钟" in initial
    assert "token 预算：无限" in initial
    assert tracker.started_at.isoformat(timespec="seconds") in initial
    assert tracker.reminders() == []
    assert tracker.reminders(10**30) == []
    for percent in (50, 60, 70, 80, 90, 95):
        clock[0] = 100 + 2400 * percent / 100 - 0.01
        assert tracker.reminders() == []
        clock[0] += 0.01
        messages = tracker.reminders()
        assert len(messages) == 1
        assert messages[0]["role"] == "system"
        assert f"{percent}%" in messages[0]["content"]
        assert "当前时间" in messages[0]["content"]
        remaining = round(2400 * (100 - percent) / 100)
        assert f"剩余 {remaining // 60} 分 {remaining % 60} 秒" in messages[0]["content"]
        assert tracker.reminders() == []
    assert tracker.initial_prompt() == initial
    tracker = budget.TimeBudget(60)
    clock[0] += 70
    messages = tracker.reminders()
    assert len(messages) == 6
    assert all("剩余 0 分 0 秒" in message["content"] for message in messages)
    assert tracker.reminders() == []


def test_token_thresholds_trigger_independently(monkeypatch):
    """时间和 token 可分别触发同一阈值，每条提醒均包含两项预算。"""
    clock = [0.0]
    monkeypatch.setattr(budget.time, "monotonic", lambda: clock[0])
    tracker = budget.TimeBudget(100, tokens=1_000)
    assert tracker.reminders(499) == []
    token_message, = tracker.reminders(500)
    assert token_message["role"] == "system"
    assert "token达到 50%" in token_message["content"]
    assert "剩余 1 分 40 秒" in token_message["content"]
    assert "剩余 token 500" in token_message["content"]

    clock[0] = 50
    time_message, = tracker.reminders(500)
    assert "时间达到 50%" in time_message["content"]
    assert "剩余 token 500" in time_message["content"]
    assert tracker.reminders(500) == []

    messages = tracker.reminders(1_100)
    assert len(messages) == 5
    assert all("剩余 token 0" in message["content"] for message in messages)
    assert tracker.reminders(1_100) == []


@pytest.mark.parametrize("tokens", [0, -1, 1.5, float("inf"), True, "1000"])
def test_invalid_token_budget_config(tokens):
    """有限的 token 预算必须使用正整数。"""
    with pytest.raises(ValueError):
        TimeBudgetConfig(tokens=tokens)
    with pytest.raises(ValueError):
        RuntimeSettings(router_token_budget=tokens)


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf"), True, "2400"])
def test_invalid_budget_config(seconds):
    """内核和应用配置均拒绝无效预算。"""
    with pytest.raises(ValueError):
        TimeBudgetConfig(seconds=seconds)
    with pytest.raises(ValueError):
        RuntimeSettings(router_time_budget=seconds)


@pytest.mark.parametrize("worker", [False, True])
@pytest.mark.parametrize("compact", [False, True])
def test_loop_budget_preserves_tool_batch_and_resets_per_run(
    monkeypatch, tmp_path, worker, compact,
):
    """长工具批次后补齐阈值；每轮重置，Worker 不独立提醒。"""
    clock = [0.0]
    monkeypatch.setattr(budget.time, "monotonic", lambda: clock[0])
    state = SessionState(
        session_id="s", agent_id="a", task_id="t", workdir=tmp_path,
        parent_agent_id="parent" if worker else None,
    )
    loop = AgentLoop(
        Model(ModelConfig(model="openai/fixture", api_key="fixture")), state,
        time_budget_config=TimeBudgetConfig(seconds=100, tokens=100),
        history_config=HistoryConfig(context_window_tokens=100, keep_recent_turns=0),
    )
    requests = []

    def call(name, identifier):
        return {
            "id": identifier, "type": "function",
            "function": {"name": name, "arguments": '{"content":"完成"}'},
        }

    async def request(messages, options):
        if messages[-1].get("content") == COMPACT_PROMPT:
            message = litellm.Message(content="工具批次已完成。")
            usage = SimpleNamespace(total_tokens=None, prompt_tokens=6, completion_tokens=4)
        else:
            requests.append(deepcopy(messages))
            calls = (
                [call("advance", "a"), call("advance", "b")]
                if len(requests) == 1 else [call("finish", "done")]
            )
            message = litellm.Message(tool_calls=calls)
            usage = SimpleNamespace(total_tokens=60) if len(requests) == 1 else None
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)

    def advance(content):
        clock[0] += 36
        return content

    monkeypatch.setattr(loop.model, "_request_with_retries", request)
    monkeypatch.setattr(
        loop.model, "count_tokens",
        lambda messages, **kwargs: (
            90 if compact and requests and not any(
                str(message.get("content", "")).startswith(SUMMARY_PREFIX)
                for message in messages
            ) else 10
        ),
    )
    loop.tools.register_callable("advance", advance, description="推进模拟时钟")
    assert asyncio.run(loop.run("开始")) == "完成"
    reminders = [message for message in state.messages if message["role"] == "system"]
    assert len(reminders) == (0 if worker else 3)
    assert ("时间预算：" in requests[0][0]["content"]) is not worker
    if not worker:
        if compact:
            assert requests[1][1]["content"].startswith(SUMMARY_PREFIX)
            assert requests[1][-3:] == reminders
        else:
            assert [message["role"] for message in requests[1][-5:]] == [
                "tool", "tool", "system", "system", "system",
            ]
        assert "剩余 0 分 28 秒" in reminders[-1]["content"]
        assert "1.66667 分钟" in requests[0][0]["content"]
        assert "token 预算：100" in requests[0][0]["content"]
        assert "时间、token达到 50%" in reminders[0]["content"]
        expected_trigger = "时间、token" if compact else "时间"
        assert f"{expected_trigger}达到 70%" in reminders[-1]["content"]
        assert f"剩余 token {30 if compact else 40}" in reminders[-1]["content"]
    assert asyncio.run(loop.run("下一轮")) == "完成"
    restored = SessionState.model_validate_json(state.model_dump_json())
    assert restored.messages == state.messages
    assert len([m for m in state.messages if m["role"] == "system"]) == len(reminders)
    if not worker:
        assert loop.time_budget.started == 72
    assert TimeBudgetConfig().seconds == RuntimeSettings().router_time_budget == 2400
