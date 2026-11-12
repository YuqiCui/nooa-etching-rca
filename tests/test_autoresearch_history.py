"""验证压缩的消息边界、完整记录、失败回滚和会话续接。"""

import asyncio
import json
from copy import deepcopy

import litellm
import pytest
from pydantic import ValidationError

from factor_demo.autoresearch_agent_v2.agent import Session
from factor_demo.autoresearch_agent_v2.agent.history import (
    COMPACT_PROMPT,
    SUMMARY_PREFIX,
    compact_history,
    get_context_messages,
)
from factor_demo.autoresearch_agent_v2.agent.model import Model
from factor_demo.autoresearch_agent_v2.agent.typing import SessionState
from factor_demo.autoresearch_agent_v2.config import ModelConfig


def make_state(tmp_path):
    """构造包含两轮已完成调用和签名推理的历史。"""
    messages = []
    for index in range(2):
        messages.extend(
            [
                {"role": "user", "content": f"任务{index} " + "原始输入 " * 300},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "旧推理", "signature": "旧签名"},
                        {"type": "text", "text": "检查完成"},
                    ],
                    "reasoning_content": "完整推理 " * 100,
                    "thinking_blocks": [{"signature": "签名"}],
                    "tool_calls": [
                        {
                            "id": f"call-{index}",
                            "type": "function",
                            "function": {"name": "inspect", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": f"call-{index}", "content": "结果 " * 300},
            ]
        )
    return SessionState(
        session_id="s1", agent_id="a1", task_id="t1", workdir=tmp_path, messages=messages
    )


def make_model():
    """使用模拟服务所需的模型配置，不读取凭据。"""
    return Model(ModelConfig(model="openai/fixture-model", api_key="fixture-key"))


def test_compact_preserves_archive_and_replays_summary_with_complete_recent_turn(
    tmp_path, monkeypatch
):
    state = make_state(tmp_path)
    original = state.get_messages()
    requests = []

    async def complete(**kwargs):
        requests.append(deepcopy(kwargs))
        return litellm.ModelResponse(
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "content": "第一轮已检查，下一步继续第二轮。",
                    }
                }
            ]
        )

    monkeypatch.setattr(litellm, "acompletion", complete)
    assert asyncio.run(compact_history(state, make_model(), system_prompt="当前工具说明", keep_recent_turns=1))
    assert state.get_messages() == original
    assert state.compaction.end_index == 3
    assert requests[0]["messages"][0]["content"] == COMPACT_PROMPT
    assert "tools" not in requests[0] and "tool_choice" not in requests[0]
    material = json.loads(requests[0]["messages"][1]["content"])
    assert material == {"system_prompt": "当前工具说明", "history": original[:3]}

    context = get_context_messages(state)
    assert context[0]["content"] == SUMMARY_PREFIX + state.compaction.summary
    assert context[1] == original[3]
    assert context[2]["tool_calls"] == original[4]["tool_calls"]
    assert context[3] == original[5]
    assert "reasoning_content" not in context[2]
    assert "thinking_blocks" not in context[2]
    assert context[2]["content"] == [{"type": "text", "text": "检查完成"}]
    context[2]["tool_calls"][0]["id"] = "changed"
    assert state.get_messages() == original

    restored = SessionState.model_validate_json(state.model_dump_json())
    assert get_context_messages(restored) == get_context_messages(state)
    restored.messages.append(
        {"role": "assistant", "content": "新内容", "reasoning_content": "新推理"}
    )
    assert get_context_messages(restored)[-1]["reasoning_content"] == "新推理"


def test_repeated_compaction_merges_previous_summary_and_only_new_prefix(tmp_path, monkeypatch):
    state = make_state(tmp_path)
    requests = []

    async def complete(**kwargs):
        requests.append(deepcopy(kwargs))
        return litellm.ModelResponse(
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "content": f"合并摘要{len(requests)}。",
                    }
                }
            ]
        )

    monkeypatch.setattr(litellm, "acompletion", complete)
    model = make_model()

    async def run():
        assert await compact_history(state, model, system_prompt="系统", keep_recent_turns=1)
        assert not await compact_history(state, model, system_prompt="系统", keep_recent_turns=1)
        state.messages.append({"role": "user", "content": "继续检查第三轮"})
        assert await compact_history(state, model, system_prompt="系统", keep_recent_turns=1)

    asyncio.run(run())
    material = json.loads(requests[1]["messages"][1]["content"])["history"]
    assert material[0]["content"] == SUMMARY_PREFIX + "合并摘要1。"
    assert len(material) == 4
    assert state.compaction.end_index == 6
    assert get_context_messages(state)[-1] == state.messages[-1]
    assert len(state.messages) == 7


@pytest.mark.parametrize("failure", ["empty", "tool", "too_long", "error", "cancel"])
def test_invalid_or_failed_compaction_preserves_previous_checkpoint(tmp_path, monkeypatch, failure):
    state = make_state(tmp_path)

    async def complete(**kwargs):
        return litellm.ModelResponse(
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "content": "首轮摘要。",
                    }
                }
            ]
        )

    monkeypatch.setattr(litellm, "acompletion", complete)
    model = make_model()
    assert asyncio.run(compact_history(state, model, system_prompt="系统", keep_recent_turns=1))
    before = state.model_dump_json()

    async def fail(**kwargs):
        if failure == "error":
            raise RuntimeError("服务不可用")
        if failure == "cancel":
            raise asyncio.CancelledError
        return litellm.ModelResponse(
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "content": "过长摘要 " * 5000 if failure == "too_long" else " ",
                        "tool_calls": [
                            {
                                "id": "unexpected",
                                "type": "function",
                                "function": {"name": "finish", "arguments": "{}"},
                            }
                        ]
                        if failure == "tool"
                        else None,
                    }
                }
            ]
        )

    monkeypatch.setattr(litellm, "acompletion", fail)
    error_type = {"error": RuntimeError, "cancel": asyncio.CancelledError}.get(failure, ValueError)
    with pytest.raises(error_type):
        asyncio.run(compact_history(state, model, system_prompt="系统", keep_recent_turns=0))
    assert state.model_dump_json() == before


@pytest.mark.parametrize("mutation", ["pending", "orphan", "duplicate", "interrupted"])
def test_incomplete_tool_protocol_is_rejected_before_model_request(tmp_path, monkeypatch, mutation):
    state = make_state(tmp_path)
    if mutation == "pending":
        state.messages.pop()
    elif mutation == "orphan":
        state.messages[-1]["tool_call_id"] = "unknown"
    elif mutation == "duplicate":
        calls = state.messages[-2]["tool_calls"]
        calls.append(deepcopy(calls[0]))
    else:
        state.messages.insert(-1, {"role": "user", "content": "在工具批次中插入消息"})

    async def unexpected(**kwargs):
        pytest.fail("工具协议不完整时不能请求模型")

    monkeypatch.setattr(litellm, "acompletion", unexpected)
    before = state.model_dump_json()
    with pytest.raises(ValueError):
        asyncio.run(compact_history(state, make_model(), system_prompt="系统", keep_recent_turns=0))
    assert state.model_dump_json() == before


def test_concurrent_history_change_does_not_publish_stale_summary(tmp_path, monkeypatch):
    state = make_state(tmp_path)

    async def complete(**kwargs):
        state.messages.append({"role": "user", "content": "并发补充"})
        return litellm.ModelResponse(
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "content": "旧快照摘要。",
                    }
                }
            ]
        )

    monkeypatch.setattr(litellm, "acompletion", complete)
    with pytest.raises(RuntimeError, match="历史已改变"):
        asyncio.run(compact_history(state, make_model(), system_prompt="系统", keep_recent_turns=1))
    assert state.compaction is None
    assert state.messages[-1]["content"] == "并发补充"


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_invalid_retention_is_rejected(tmp_path, value):
    with pytest.raises(ValueError, match="保留轮数"):
        asyncio.run(
            compact_history(
                make_state(tmp_path),
                make_model(),
                system_prompt="系统",
                keep_recent_turns=value,
            )
        )


def test_session_continues_after_restore(tmp_path, monkeypatch, capsys):
    state = make_state(tmp_path)
    original = state.get_messages()
    requests = []

    async def complete(**kwargs):
        requests.append(deepcopy(kwargs))
        if len(requests) == 1:
            message = {"role": "assistant", "content": "保留任务与已验证结果，继续检查。"}
        else:
            message = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "done",
                        "type": "function",
                        "function": {"name": "finish", "arguments": '{"content":"完成"}'},
                    }
                ],
            }
        return litellm.ModelResponse(choices=[{"message": message}])

    monkeypatch.setattr(litellm, "acompletion", complete)

    async def run():
        async with Session(make_model(), state) as session:
            assert await session.compact(keep_recent_turns=0)
            assert "compact" not in [s["function"]["name"] for s in session.schemas()]
        restored = SessionState.model_validate_json(state.model_dump_json())
        async with Session(make_model(), restored) as session:
            assert await session.run("继续") == "完成"
        assert restored.get_messages()[: len(original)] == original
        assert requests[1]["messages"][1]["content"].startswith(SUMMARY_PREFIX)
        assert requests[1]["messages"][2] == {"role": "user", "content": "继续"}

    asyncio.run(run())
    assert capsys.readouterr().out == "完成\n"


def test_checkpoint_bounds_are_validated_on_restore(tmp_path):
    data = make_state(tmp_path).model_dump()
    data["compaction"] = {"summary": "摘要", "end_index": 3, "source_message_count": 99}
    with pytest.raises(ValidationError, match="超出消息历史范围"):
        SessionState.model_validate(data)
