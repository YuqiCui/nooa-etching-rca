"""验证会话隔离、完整历史和连续多轮执行。"""

import asyncio
import json
from copy import deepcopy

import litellm
import pytest

from factor_demo.autoresearch_agent_v2.agent import Session
from factor_demo.autoresearch_agent_v2.agent.model import Model
from factor_demo.autoresearch_agent_v2.agent.typing import SessionState
from factor_demo.autoresearch_agent_v2.config import ModelConfig


def test_same_task_sessions_share_workdir_but_keep_independent_messages(tmp_path):
    main = SessionState(
        session_id="session-main", agent_id="main", task_id="task-1", workdir=tmp_path
    )
    child = SessionState(
        session_id="session-child",
        agent_id="child",
        parent_agent_id=main.agent_id,
        task_id=main.task_id,
        workdir=main.workdir,
    )

    main.messages.append({"role": "user", "content": "Main task"})
    child.messages.append({"role": "user", "content": "Child task"})

    assert main.workdir == child.workdir
    assert main.get_messages() == [{"role": "user", "content": "Main task"}]
    assert child.get_messages() == [{"role": "user", "content": "Child task"}]


def test_complete_messages_survive_reading_and_json_roundtrip(tmp_path):
    messages = [
        {"role": "user", "content": "Full input " * 5000},
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "Complete reasoning " * 5000,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "inspect", "arguments": '{"value": 1}'},
                }
            ],
            "provider_extension": {"opaque": ["preserve", {"signature": "signature-1"}]},
        },
        {"role": "tool", "tool_call_id": "call-1", "content": '{"value": 1}'},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Full thinking", "signature": "signature-2"},
                {"type": "redacted_thinking", "data": "opaque-data"},
                {"type": "text", "text": "Result"},
            ],
        },
    ]
    session = SessionState(
        session_id="session-1",
        agent_id="main",
        task_id="task-1",
        workdir=tmp_path,
        messages=messages,
    )

    assert session.get_messages() == messages
    restored = SessionState.model_validate_json(session.model_dump_json())
    assert restored.get_messages() == messages


def test_returned_history_cannot_mutate_stored_nested_messages(tmp_path):
    original = [
        {
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": "Reason", "signature": "original"}],
        }
    ]
    session = SessionState(
        session_id="session-1",
        agent_id="main",
        task_id="task-1",
        workdir=tmp_path,
        messages=deepcopy(original),
    )

    returned = session.get_messages()
    returned[0]["content"][0]["signature"] = "changed"
    returned.append({"role": "user", "content": "Another message"})

    assert session.messages == original


@pytest.mark.parametrize("with_skipped_calls", [False, True])
def test_session_continues_after_finish_with_complete_tool_history(
    monkeypatch, tmp_path, capsys, with_skipped_calls
):
    requests = []

    def call(name, content, call_id):
        return {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps({"content": content})},
        }

    async def complete(**kwargs):
        requests.append(deepcopy(kwargs))
        turn = len(requests)
        calls = [call("finish", f"结果{turn}", f"finish-{turn}")]
        if turn == 1 and with_skipped_calls:
            calls.extend(
                [
                    call("inspect", "不应打印", "skipped-1"),
                    {
                        "id": "skipped-2",
                        "type": "function",
                        "function": {
                            "name": "unknown",
                            "arguments": "invalid json",
                        },
                    },
                ]
            )
        return litellm.ModelResponse(
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "reasoning_content": f"推理{turn}",
                        "tool_calls": calls,
                    }
                }
            ]
        )

    monkeypatch.setattr(litellm, "acompletion", complete)
    model = Model(ModelConfig(model="openai/fixture-model", api_key="fixture-key"))
    state = SessionState(session_id="s1", agent_id="a1", task_id="t1", workdir=tmp_path)
    session = Session(model, state, system_prompt="系统提示词")
    loop = session.loop

    async def converse():
        assert await session.run("第一条消息") == "结果1"
        first_history = state.get_messages()
        assert await session.run("继续分析") == "结果2"
        assert requests[1]["messages"] == [
            {"role": "system", "content": "系统提示词"},
            *first_history,
            {"role": "user", "content": "继续分析"},
        ]
        assert session.loop is loop
        assert loop.session is state

        restored_state = SessionState.model_validate_json(state.model_dump_json())
        restored = Session(model, restored_state, system_prompt="系统提示词")
        assert await restored.run("再解释一下") == "结果3"
        assert requests[2]["messages"][1:-1] == state.get_messages()

    asyncio.run(converse())
    assert capsys.readouterr().out == "结果1\n结果2\n结果3\n"
    assert state.messages[1]["reasoning_content"] == "推理1"
    for request in requests[1:]:
        pending = []
        for message in request["messages"]:
            if message["role"] == "assistant":
                assert pending == []
                pending = [item["id"] for item in message["tool_calls"]]
            elif message["role"] == "tool":
                assert message["tool_call_id"] == pending.pop(0)
            else:
                assert pending == []
        assert pending == []
    if with_skipped_calls:
        skipped = [m for m in state.messages if m.get("tool_call_id", "").startswith("skipped-")]
        assert [m["tool_call_id"] for m in skipped] == ["skipped-1", "skipped-2"]
        assert all("未执行" in m["content"] for m in skipped)
