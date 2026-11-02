"""Model adapter checks using real SDK requests and simulated provider responses."""

import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace

import httpx
import litellm
import pytest

from factor_demo.autoresearch_agent_v2.agent.model import Model
from factor_demo.autoresearch_agent_v2.agent.typing import SessionState
from factor_demo.autoresearch_agent_v2.config import ModelConfig


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_completion_preserves_thinking_and_tool_calls_across_requests(
    provider, monkeypatch, tmp_path
):
    captured = []
    reasoning = "Complete reasoning " * 2000
    text = '{"name":"not_a_call","arguments":{}}'
    tools = [
        {
            "type": "function",
            "function": {
                "name": "inspect",
                "description": "检查样本",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                    "required": ["value"],
                },
            },
        }
    ]
    tool_call = {
        "id": "call-1",
        "type": "function",
        "function": {"name": "inspect", "arguments": '{"value":1}'},
    }

    async def send(client, request, **kwargs):
        assert request.url.host == f"{provider}.invalid"
        body = json.loads(request.content)
        captured.append(body)
        if provider == "openai":
            assert request.url.path == "/v1/chat/completions"
            assert request.headers["authorization"] == "Bearer fixture-key"
            payload = {
                "id": "completion-1",
                "object": "chat.completion",
                "created": 0,
                "model": "fixture-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": text,
                            "reasoning_content": reasoning,
                            "tool_calls": [tool_call],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            }
        else:
            assert request.url.path == "/v1/messages"
            assert request.headers["x-api-key"] == "fixture-key"
            payload = {
                "id": "message-1",
                "type": "message",
                "role": "assistant",
                "model": "fixture-model",
                "content": [
                    {"type": "thinking", "thinking": reasoning, "signature": "signature-1"},
                    {"type": "redacted_thinking", "data": "opaque-data"},
                    {"type": "text", "text": text},
                    {"type": "tool_use", "id": "call-1", "name": "inspect", "input": {"value": 1}},
                ],
                "stop_reason": "tool_use",
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 20},
            }
        return httpx.Response(200, request=request, json=payload)

    monkeypatch.setattr(httpx.AsyncClient, "send", send)

    async def exercise():
        session = SessionState(
            session_id="session-1",
            agent_id="agent-1",
            task_id="task-1",
            workdir=tmp_path,
            messages=[{"role": "user", "content": "Inspect the sample."}],
        )
        model = Model(
            ModelConfig(
                model=f"{provider}/fixture-model",
                api_key="fixture-key",
                base_url=f"https://{provider}.invalid/v1"
                if provider == "openai"
                else f"https://{provider}.invalid",
                max_tokens=8192 if provider == "anthropic" else None,
                thinking={"type": "enabled", "budget_tokens": 1024}
                if provider == "anthropic"
                else {"type": "enabled"},
            )
        )
        messages = session.get_messages()
        original = deepcopy(messages)
        result = await model.completion(messages, tools=tools)
        assert messages == original
        assert session.messages == original
        assert isinstance(result, dict)
        assert result["role"] == "assistant"
        assert result["content"] == text
        assert result["reasoning_content"] == reasoning
        assert result["tool_calls"][0]["id"] == "call-1"
        assert json.loads(result["tool_calls"][0]["function"]["arguments"]) == {"value": 1}
        if provider == "anthropic":
            assert result["thinking_blocks"] == [
                {"type": "thinking", "thinking": reasoning, "signature": "signature-1"},
                {"type": "redacted_thinking", "data": "opaque-data"},
            ]
        session.messages.extend(
            [result, {"role": "tool", "tool_call_id": "call-1", "content": '{"value":1}'}]
        )
        await model.completion(session.get_messages(), tools=tools)

    asyncio.run(exercise())

    assert len(captured) == 2
    if provider == "openai":
        assert captured[0]["tools"] == tools
        assert "max_tokens" not in captured[0]
        assert "max_completion_tokens" not in captured[0]
        assert captured[0]["thinking"] == {"type": "enabled"}
        replayed = captured[1]["messages"][1]
        assert replayed["reasoning_content"] == reasoning
        assert replayed["tool_calls"][0] == tool_call
    else:
        assert captured[0]["tools"][0]["name"] == "inspect"
        assert captured[0]["tools"][0]["input_schema"] == tools[0]["function"]["parameters"]
        assert captured[0]["max_tokens"] == 8192
        assert captured[0]["thinking"] == {"type": "enabled", "budget_tokens": 1024}
        replayed = captured[1]["messages"][1]["content"]
        assert replayed[:2] == [
            {"type": "thinking", "thinking": reasoning, "signature": "signature-1"},
            {"type": "redacted_thinking", "data": "opaque-data"},
        ]
        assert replayed[2] == {"type": "text", "text": text}
        assert replayed[3] == {
            "type": "tool_use",
            "id": "call-1",
            "name": "inspect",
            "input": {"value": 1},
        }
        assert captured[1]["messages"][2]["content"][0]["type"] == "tool_result"
        assert captured[1]["messages"][2]["content"][0]["tool_use_id"] == "call-1"


def test_backend_mutation_does_not_change_input_history(monkeypatch):
    messages = [{"role": "assistant", "content": [{"type": "text", "text": "Original"}]}]
    original = deepcopy(messages)

    async def complete(**kwargs):
        kwargs["messages"][0]["content"][0]["text"] = "Changed"
        return litellm.ModelResponse(
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "content": "Result",
                        "provider_extension": {"signature": "opaque-value"},
                    }
                }
            ]
        )

    monkeypatch.setattr(litellm, "acompletion", complete)
    result = asyncio.run(
        Model(ModelConfig(model="openai/fixture-model", api_key="fixture-key")).completion(messages)
    )

    assert result["content"] == "Result"
    assert result["provider_extension"] == {"signature": "opaque-value"}
    assert messages == original


def test_provider_error_propagates_without_extra_adapter_retry(monkeypatch):
    error = litellm.AuthenticationError(
        message="Invalid fixture key", llm_provider="openai", model="fixture-model"
    )
    calls = []

    async def complete(**kwargs):
        calls.append(kwargs)
        raise error

    monkeypatch.setattr(litellm, "acompletion", complete)
    model = Model(ModelConfig(model="openai/fixture-model", api_key="fixture-key"))

    with pytest.raises(litellm.AuthenticationError) as caught:
        asyncio.run(model.completion([{"role": "user", "content": "Hello"}]))

    assert caught.value is error
    assert len(calls) == 1
    assert calls[0]["num_retries"] == 3


@pytest.mark.parametrize("max_tokens,num_retries", [(None, 3), (None, 10), (8192, 0), (1024, 5)])
def test_completion_forwards_configured_limits(monkeypatch, max_tokens, num_retries):
    captured = []

    async def complete(**kwargs):
        captured.append(kwargs)
        return litellm.ModelResponse(
            choices=[{"message": {"role": "assistant", "content": "Result"}}]
        )

    monkeypatch.setattr(litellm, "acompletion", complete)
    config = ModelConfig(model="openai/fixture-model", api_key="fixture-key")
    if max_tokens is not None or num_retries != 3:
        config = ModelConfig(
            model="openai/fixture-model",
            api_key="fixture-key",
            max_tokens=max_tokens,
            num_retries=num_retries,
        )

    asyncio.run(Model(config).completion([{"role": "user", "content": "Hello"}]))

    assert captured[0]["num_retries"] == num_retries
    if max_tokens is None:
        assert "max_tokens" not in captured[0]
    else:
        assert captured[0]["max_tokens"] == max_tokens


def test_empty_response_raises_clear_error(monkeypatch):
    async def complete(**kwargs):
        return SimpleNamespace(choices=[])

    monkeypatch.setattr(litellm, "acompletion", complete)
    model = Model(ModelConfig(model="openai/fixture-model", api_key="fixture-key"))

    with pytest.raises(ValueError, match="no message choice"):
        asyncio.run(model.completion([{"role": "user", "content": "Hello"}]))


@pytest.mark.parametrize("choice", ["auto", "required"])
def test_completion_forwards_explicit_tool_choice(monkeypatch, choice):
    captured = []

    async def complete(**kwargs):
        captured.append(kwargs)
        return litellm.ModelResponse(
            choices=[{"message": {"role": "assistant", "content": "Result"}}]
        )

    monkeypatch.setattr(litellm, "acompletion", complete)
    config = ModelConfig(model="openai/test", api_key="fixture", tool_choice=choice)
    tools = [{"type": "function", "function": {"name": "finish", "parameters": {}}}]
    asyncio.run(Model(config).completion([{"role": "user", "content": "task"}], tools=tools))
    assert captured[0]["tool_choice"] == choice
    assert captured[0]["tools"] == tools


def test_config_does_not_display_api_key():
    config = ModelConfig(model="openai/fixture-model", api_key="private-fixture-key")

    assert "private-fixture-key" not in repr(config)
    assert "private-fixture-key" not in config.model_dump_json()


@pytest.mark.parametrize(
    "usage", [None, {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}]
)
def test_usage_callback_receives_provider_usage_without_changing_message(monkeypatch, usage):
    received = []

    async def complete(**kwargs):
        response = litellm.ModelResponse(
            choices=[{"message": {"role": "assistant", "content": "Result"}}],
            usage=usage,
        )
        return response

    monkeypatch.setattr(litellm, "acompletion", complete)
    model = Model(
        ModelConfig(model="openai/fixture-model", api_key="fixture-key"),
        usage_callback=received.append,
    )
    result = asyncio.run(model.completion([{"role": "user", "content": "Hello"}]))
    assert result["content"] == "Result"
    assert "usage" not in result
    if usage is None:
        assert received == [None]
    else:
        assert len(received) == 1
        assert received[0]["total_tokens"] == 15
