import asyncio
import json

import httpx
import pytest

from factor_demo.providers import Provider
from factor_demo.runtime import (
    RuntimeSettings,
    build_client,
    context_truncation,
    install_summarizer,
)


def test_context_reserves_output_and_summary_headroom():
    settings = RuntimeSettings()
    assert settings.input_budget == 222_000
    assert settings.compression_threshold == 177_600
    with pytest.raises(ValueError, match="smaller"):
        RuntimeSettings(max_output_tokens=272_000)
    with pytest.raises(ValueError, match="usable input"):
        RuntimeSettings(summary_threshold=225_000)


def test_glm_cannot_silently_disable_thinking():
    p = Provider("glm", "glm-5.3-flash", "https://example.invalid/v1", "openai", "test-only")
    with pytest.raises(ValueError, match="cannot disable"):
        build_client(p, RuntimeSettings(thinking=False))


def test_real_nooa_wire_v2_output_cap_and_reasoning(monkeypatch):
    """Exercise real NOOA + SDK request construction, with a fake HTTP response."""
    captured = []

    async def send(client, request, **kwargs):
        assert request.url.host == "wire.invalid"
        body = json.loads(request.content)
        captured.append(body)
        code = "return_result({'selected_experiment_id':'exp_001','summary':'ok','limitations':[]})"
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "test-1",
                "object": "chat.completion",
                "created": 0,
                "model": "glm-5.3-flash",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "cell-1",
                                    "type": "function",
                                    "function": {
                                        "name": "python_cell",
                                        "arguments": json.dumps({"code": code}),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            },
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", send)

    async def exercise():
        from factor_demo.legacy.agent import FactorResearchAgent

        p = Provider("glm", "glm-5.3-flash", "https://wire.invalid/v1", "openai", "test-only")
        settings = RuntimeSettings()
        client = build_client(p, settings)
        assert client.context_window == 272_000
        assert client.get_context_limits().usable_input_tokens == 222_000
        agent = FactorResearchAgent(None, llm=client, truncation=context_truncation(settings))
        assert agent.runtime.truncation_config.max_context_tokens == 222_000
        summarizer = install_summarizer(agent, settings)
        assert summarizer.config.max_tokens == 177_600
        try:
            result = await agent.research("Protocol unit test only")
            assert result.selected_experiment_id == "exp_001"
        finally:
            await agent.aclose()

    asyncio.run(exercise())
    assert len(captured) == 1
    body = captured[0]
    assert body.get("max_tokens", body.get("max_completion_tokens")) == 50_000
    assert "context_window" not in body  # local budget metadata must not leak to the API
    assert body["reasoning_effort"] == "medium"
    assert body["thinking"] == {"type": "enabled", "clear_thinking": False}
    assert [t["function"]["name"] for t in body["tools"]] == ["python_cell"]


def test_anthropic_gateway_keeps_thinking_and_omits_internal_routing_ids(tmp_path, monkeypatch):
    from factor_demo.budget import MissionBudget
    from factor_demo.capabilities import Capabilities
    from factor_demo.live_events import LiveEvents

    captured = []

    async def send(client, request, **kwargs):
        assert request.url.host == "wire.invalid"
        captured.append((json.loads(request.content), dict(request.headers)))
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "msg_fixture",
                "type": "message",
                "role": "assistant",
                "model": "MiniMax-M3.1-Flash-Preview",
                "content": [{"type": "text", "text": "OK"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 2},
            },
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    events = LiveEvents(tmp_path, "wire")
    budget = MissionBudget(tmp_path, "wire", Capabilities())

    async def run():
        settings = RuntimeSettings()
        client = budget.client(
            build_client(
                Provider(
                    "fixture",
                    "MiniMax-M3.1-Flash-Preview",
                    "https://wire.invalid",
                    "anthropic",
                    "test-only",
                ),
                settings,
            ),
            events,
            settings,
        )
        try:
            await client.acall(
                [{"role": "user", "content": "Reply OK"}], prompt_cache_key="INTERNAL_AGENT_ID"
            )
        finally:
            await client.aclose()

    try:
        asyncio.run(run())
    finally:
        budget.close()
        events.close()
    assert len(captured) == 1
    body, headers = captured[0]
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 32000}
    assert body["max_tokens"] == 50000
    assert "prompt_cache_key" not in body and "x-session-affinity" not in headers
    assert "INTERNAL_AGENT_ID" not in json.dumps((body, headers))


def test_json_gateway_preserves_strict_review_contract_and_native_text():
    from types import SimpleNamespace

    from nooa.llm_types import AssistantText
    from nooa.unifiedllm import LLMResponse
    from pydantic import ValidationError

    from factor_demo.provider_compat import install_json_transport
    from factor_demo.research_models import ReviewResult

    calls = []

    async def base(messages, **kwargs):
        calls.append((messages, kwargs))
        content = (
            '{"issues":[],"strengths":[],"limitations":[]}'
            if len(calls) == 1
            else '{"issues":[{"category":"evidence","severity":"warning","claim_id":"summary","evidence_ids":{"item":"e1"},"problem":"x","counterevidence":"x","suggested_check":"x"}]}'
        )
        return LLMResponse(parts=(AssistantText(text=content),))

    client = install_json_transport(SimpleNamespace(acall=base))

    async def run():
        response = await client.acall(
            [{"role": "user", "content": "review"}], output_model=ReviewResult
        )
        assert response.parsed == ReviewResult()
        assert "output_model" not in calls[0][1] and calls[0][1]["tools"] is None
        assert "JSON Schema" in calls[0][0][-1]["content"]
        with pytest.raises(ValidationError):
            await client.acall([], output_model=ReviewResult)

    asyncio.run(run())
