"""验证自动压缩的配置、token 阈值和长工具循环续接。"""

import asyncio
import json
from copy import deepcopy

import litellm
import pytest
from pydantic import ValidationError

from factor_demo.autoresearch_agent_v2.agent import Session
from factor_demo.autoresearch_agent_v2.agent.history import COMPACT_PROMPT, SUMMARY_PREFIX
from factor_demo.autoresearch_agent_v2.agent.model import Model
from factor_demo.autoresearch_agent_v2.agent.typing import SessionState
from factor_demo.autoresearch_agent_v2.config import HistoryConfig, ModelConfig


def make_session(tmp_path, *, messages=None, **options):
    """创建使用虚构凭据和可指定历史策略的会话。"""
    return Session(
        Model(ModelConfig(model="openai/fixture-model", api_key="fixture-key")),
        SessionState(
            session_id="s1",
            agent_id="a1",
            task_id="t1",
            workdir=tmp_path,
            messages=messages or [],
        ),
        **options,
    )


def tool_reply(name, call_id):
    """生成统一 tool_calls 格式的模拟工具请求。"""
    arguments = {"content": "完成" if name == "finish" else "采样"}
    return {
        "role": "assistant",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


@pytest.mark.parametrize(
    "tokens,percent,expected",
    [
        (7999, 80, False),
        (8000, 80, True),
        (8001, 80, True),
        (7000, 65, True),
        (7000, 80, False),
    ],
)
def test_threshold_uses_config_and_actual_request_before_completion(
    tmp_path, monkeypatch, tokens, percent, expected
):
    counts = []
    requests = []
    original = [
        {"role": "user", "content": "原任务 " * 1000},
        {
            "role": "assistant",
            "content": "已完成检查 " * 1000,
        },
    ]
    session = make_session(
        tmp_path,
        messages=deepcopy(original),
        history_config=HistoryConfig(context_window_tokens=10_000, compact_trigger_percent=percent),
    )
    session.register_dataset("样本", "sample.csv", "登记数据集用于分析")

    def count(**kwargs):
        counts.append(deepcopy(kwargs))
        compacted = any(
            isinstance(message.get("content"), str)
            and message["content"].startswith(SUMMARY_PREFIX)
            for message in kwargs["messages"]
        )
        return 2000 if compacted else tokens

    async def complete(**kwargs):
        requests.append(deepcopy(kwargs))
        message = (
            {"role": "assistant", "content": "原任务已检查，保留已验证结论。"}
            if kwargs["messages"][0]["content"] == COMPACT_PROMPT
            else tool_reply("finish", "done")
        )
        return litellm.ModelResponse(choices=[{"message": message}])

    monkeypatch.setattr(litellm, "token_counter", count)
    monkeypatch.setattr(litellm, "acompletion", complete)

    async def run():
        async with session:
            assert await session.run("继续") == "完成"

    asyncio.run(run())
    assert bool(session.state.compaction) == expected
    assert len(requests) == (2 if expected else 1)
    assert counts[-1]["messages"] == requests[-1]["messages"]
    assert counts[-1]["tools"] == requests[-1].get("tools")
    assert counts[-1]["use_default_image_token_count"] is True
    assert "样本" in counts[0]["messages"][0]["content"]
    assert session.state.messages[:2] == original
    if expected:
        assert "tools" not in requests[0] and "tool_choice" not in requests[0]
        assert requests[-1]["messages"][-1] == {"role": "user", "content": "继续"}
    assert {item["function"]["name"] for item in counts[0]["tools"]} == {
        "finish",
        "register_dataset",
    }


def test_single_user_task_compacts_repeatedly_between_complete_tool_batches(tmp_path, monkeypatch):
    session = make_session(
        tmp_path,
        history_config=HistoryConfig(context_window_tokens=10_000),
    )
    executed = []
    summaries = []
    task_requests = []

    def inspect(content: str):
        executed.append(content)
        return "观测结果 " * 400

    session.loop.tools.register_callable("inspect", inspect, description="模拟采样")

    def count(**kwargs):
        payload = json.dumps(kwargs["messages"], ensure_ascii=False)
        return 8500 if payload.count("观测结果") >= 800 else 2000

    async def complete(**kwargs):
        if kwargs["messages"][0]["content"] == COMPACT_PROMPT:
            summaries.append(deepcopy(kwargs))
            message = {"role": "assistant", "content": "已完成早期采样，继续采样并核对结论。"}
        else:
            task_requests.append(deepcopy(kwargs))
            index = len(task_requests)
            message = tool_reply("inspect" if index < 4 else "finish", f"call-{index}")
        return litellm.ModelResponse(choices=[{"message": message}])

    monkeypatch.setattr(litellm, "token_counter", count)
    monkeypatch.setattr(litellm, "acompletion", complete)

    async def run():
        async with session:
            assert await session.run("完成长任务") == "完成"

    asyncio.run(run())
    assert executed == ["采样"] * 3
    assert len(task_requests) == 4
    assert len(summaries) == 2
    assert session.state.compaction.end_index == 5
    for request in task_requests[2:]:
        assert request["messages"][1]["content"].startswith(SUMMARY_PREFIX)
        assert request["messages"][2]["role"] == "assistant"
        assert (
            request["messages"][3]["tool_call_id"] == request["messages"][2]["tool_calls"][0]["id"]
        )
    assert session.state.messages[0]["content"] == "完成长任务"
    assert sum(message["role"] == "tool" for message in session.state.messages) == 4
    restored = SessionState.model_validate_json(session.state.model_dump_json())
    assert restored.compaction == session.state.compaction


@pytest.mark.parametrize("override,expected_end", [(None, 4), (1, 2)])
def test_manual_compact_uses_configured_retention_with_per_call_override(
    tmp_path, monkeypatch, override, expected_end
):
    messages = [
        {"role": "user", "content": "任务 " * 1000},
        {"role": "assistant", "content": "第一轮结果 " * 1000},
        {"role": "user", "content": "继续 " * 1000},
        {"role": "assistant", "content": "第二轮结果 " * 1000},
    ]
    session = make_session(
        tmp_path, messages=messages, history_config=HistoryConfig(keep_recent_turns=0)
    )

    async def complete(**kwargs):
        return litellm.ModelResponse(
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "content": "已有检查结果，继续原任务。",
                    }
                }
            ]
        )

    monkeypatch.setattr(litellm, "acompletion", complete)

    async def run():
        async with session:
            assert await session.compact(keep_recent_turns=override)

    asyncio.run(run())
    assert session.state.compaction.end_index == expected_end


@pytest.mark.parametrize("tokens", [8500, 10_000])
def test_uncompactable_input_does_not_repeat_compaction_requests(tmp_path, monkeypatch, tokens):
    session = make_session(tmp_path, history_config=HistoryConfig(context_window_tokens=10_000))
    requests = []
    monkeypatch.setattr(litellm, "token_counter", lambda **kwargs: tokens)

    async def complete(**kwargs):
        requests.append(kwargs)
        assert kwargs["messages"][0]["content"] != COMPACT_PROMPT
        return litellm.ModelResponse(choices=[{"message": tool_reply("finish", "done")}])

    monkeypatch.setattr(litellm, "acompletion", complete)

    async def run():
        async with session:
            if tokens == 10_000:
                with pytest.raises(ValueError, match="上下文 token 预算"):
                    await session.run("当前唯一输入")
            else:
                assert await session.run("当前唯一输入") == "完成"

    asyncio.run(run())
    assert len(requests) == (0 if tokens == 10_000 else 1)
    assert session.state.compaction is None


def test_failed_automatic_compaction_stops_before_task_request(tmp_path, monkeypatch):
    session = make_session(
        tmp_path,
        messages=[{"role": "user", "content": "历史 " * 1000}],
        history_config=HistoryConfig(context_window_tokens=10_000),
    )
    monkeypatch.setattr(litellm, "token_counter", lambda **kwargs: 8000)
    requests = []

    async def complete(**kwargs):
        requests.append(kwargs)
        raise RuntimeError("摘要服务失败")

    monkeypatch.setattr(litellm, "acompletion", complete)

    async def run():
        async with session:
            # 两条连续 user 消息属于同一待回复轮次，先补齐一轮历史。
            session.state.messages.append({"role": "assistant", "content": "已有结果"})
            with pytest.raises(RuntimeError, match="摘要服务失败"):
                await session.run("继续")

    asyncio.run(run())
    assert len(requests) == 1
    assert requests[0]["messages"][0]["content"] == COMPACT_PROMPT
    assert session.state.compaction is None
    assert session.state.messages[-1] == {"role": "user", "content": "继续"}


@pytest.mark.parametrize(
    "values",
    [
        {"context_window_tokens": 0},
        {"context_window_tokens": True},
        {"keep_recent_turns": -1},
        {"keep_recent_turns": True},
        {"compact_trigger_percent": 0},
        {"compact_trigger_percent": 101},
        {"compact_trigger_percent": True},
        {"compact_trigger_percent": float("nan")},
        {"compact_trigger_percent": float("inf")},
    ],
)
def test_history_config_rejects_invalid_values(values):
    with pytest.raises(ValidationError):
        HistoryConfig(**values)


def test_real_local_token_counter_includes_tools_and_reasoning_without_mutating_input(tmp_path):
    session = make_session(tmp_path)
    messages = [
        {"role": "system", "content": "系统说明"},
        {"role": "user", "content": "分析数据"},
    ]
    original = deepcopy(messages)
    base = session.loop.model.count_tokens(messages)
    assert base > 0
    assert session.loop.model.count_tokens(messages, tools=session.loop.tools.schemas()) > base
    messages.append({"role": "assistant", "content": None, "reasoning_content": "分析 " * 100})
    assert session.loop.model.count_tokens(messages) > base
    assert messages[:2] == original


def test_disabled_token_counter_is_reported_instead_of_silently_skipping_monitoring(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(litellm, "disable_token_counter", True)
    with pytest.raises(ValueError, match="token 计数不可用"):
        make_session(tmp_path).loop.model.count_tokens([{"role": "user", "content": "任务"}])
