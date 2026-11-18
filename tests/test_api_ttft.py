"""验证首个有效增量计时、消息合并和失败尝试记录。"""

import asyncio
import json
from types import SimpleNamespace

import httpx
import litellm
import pytest

from factor_demo.autoresearch_agent_v2.agent import model as adapter
from factor_demo.autoresearch_agent_v2.config import ModelConfig
from factor_demo.autoresearch_web import WebModel


def make_model(**options):
    """创建启用首输出计时的模拟服务配置。"""
    return adapter.Model(ModelConfig(
        model="openai/fixture", api_key="fixture", measure_ttft=True, num_retries=0,
        **options,
    ))


def test_web_persists_timing_with_agent_and_task(tmp_path):
    """保存每个会话的计时，并保留任务与委派关联。"""
    client = WebModel(
        ModelConfig(model="openai/fixture", api_key="fixture"),
        SimpleNamespace(agent_id="worker1", assignment_id="assignment1"),
        tmp_path / "session.json", SimpleNamespace(task_id="task1"),
    )
    client.record_request_timing({"attempt_id": "a", "event": "first_token", "ttft_s": 1.5})
    record = json.loads((tmp_path / "api_timing.jsonl").read_text())
    assert record == {
        "attempt_id": "a", "event": "first_token", "ttft_s": 1.5,
        "task_id": "task1", "agent_id": "worker1", "assignment_id": "assignment1",
    }


def chunk(delta=None, *, usage=None, finish_reason=None):
    """构造 SDK 使用的标准增量。"""
    return litellm.ModelResponseStream(
        id="fixture", model="fixture", created=0,
        choices=[{"index": 0, "delta": delta or {}, "finish_reason": finish_reason}],
        **({"usage": usage} if usage else {}),
    )


def test_first_reasoning_token_skips_role_and_preserves_tools(monkeypatch):
    """空包不计首输出，思考增量计入；工具参数与用量可完整合并。"""
    clock = [100.0]
    monkeypatch.setattr(adapter, "time", SimpleNamespace(perf_counter=lambda: clock[0]))
    closed = []

    class Stream:
        """用可控时间提供增量并记录连接关闭。"""

        async def __aiter__(self):
            for elapsed, value in [
                (1, chunk({"role": "assistant", "content": ""})),
                (2, chunk({"reasoning_content": "先分析"})),
                (3, chunk({"reasoning_content": "再行动"})),
                (4, chunk({"tool_calls": [{"index": 0, "id": "call1", "type": "function", "function": {"name": "finish", "arguments": '{"content":'}}]})),
                (5, chunk({"tool_calls": [{"index": 0, "function": {"arguments": '"完成"}'}}]}, finish_reason="tool_calls")),
                (6, chunk(usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})),
            ]:
                clock[0] = 100 + elapsed
                yield value

        async def aclose(self):
            closed.append(True)

    async def complete(**kwargs):
        assert kwargs["stream"] is True
        assert kwargs["stream_options"] == {"include_usage": True}
        return Stream()

    monkeypatch.setattr(litellm, "acompletion", complete)
    client = make_model()
    timings, usages = [], []
    client.request_timing_callback = timings.append
    client.usage_callback = usages.append
    result = asyncio.run(client.completion([{"role": "user", "content": "开始"}]))
    assert result["reasoning_content"] == "先分析再行动"
    assert result["tool_calls"][0]["function"] == {
        "name": "finish", "arguments": '{"content":"完成"}',
    }
    assert usages[0]["total_tokens"] == 15
    assert [value["event"] for value in timings] == ["first_token", "end"]
    assert timings[0]["ttft_s"] == 2
    assert timings[0]["first_token_kind"] == "reasoning_content"
    assert timings[1]["duration_s"] == 6
    assert timings[1]["status"] == "completed"
    assert timings[0]["attempt_id"] == timings[1]["attempt_id"]
    assert closed == [True]


@pytest.mark.parametrize("after_token", [False, True])
def test_timeout_records_missing_or_received_first_token_and_closes(monkeypatch, after_token):
    """首输出之前或之后超时均留下终态，不把完整耗时冒充首输出时间。"""
    closed = []

    class Stream:
        """在指定阶段模拟挂起。"""

        async def __aiter__(self):
            if after_token:
                yield chunk({"content": "开始"})
            await asyncio.Event().wait()

        async def aclose(self):
            closed.append(True)

    async def complete(**kwargs):
        return Stream()

    monkeypatch.setattr(litellm, "acompletion", complete)
    client = make_model(timeout_s=0.02)
    timings = []
    client.request_timing_callback = timings.append
    with pytest.raises(litellm.Timeout):
        asyncio.run(client.completion([]))
    final = timings[-1]
    assert final["event"] == "end" and final["error_type"] == "Timeout"
    assert (final["ttft_s"] is not None) is after_token
    assert final["status"] == "failed"
    assert closed == [True]


def test_real_openai_sdk_stream_and_absent_usage(monkeypatch):
    """通过真实 SDK 解析模拟 SSE，核对首正文及服务端缺失用量。"""
    async def send(client, request, **kwargs):
        body = json.loads(request.content)
        assert body["stream"] is True
        values = [
            {"role": "assistant"}, {"content": "你"}, {"content": "好"},
        ]
        packets = [
            {"id": "fixture", "object": "chat.completion.chunk", "created": 0,
             "model": "fixture", "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
            for delta in values
        ]
        packets.append({"id": "fixture", "object": "chat.completion.chunk", "created": 0,
                        "model": "fixture", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
        data = "".join(f"data: {json.dumps(p)}\n\n" for p in packets) + "data: [DONE]\n\n"
        return httpx.Response(200, request=request, headers={"content-type": "text/event-stream"}, content=data)

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    client = make_model(base_url="https://ttft.invalid/v1")
    timings, usages = [], []
    client.request_timing_callback = timings.append
    client.usage_callback = usages.append
    result = asyncio.run(client.completion([{"role": "user", "content": "hello"}]))
    assert result["content"] == "你好"
    assert timings[0]["first_token_kind"] == "content"
    assert 0 <= timings[-1]["ttft_s"] <= timings[-1]["duration_s"]
    assert usages == [None]


def test_real_anthropic_sdk_preserves_thinking_signature_and_tools(monkeypatch):
    """通过真实 SDK 合并 Anthropic 思考签名、正文与工具参数。"""
    async def send(client, request, **kwargs):
        assert json.loads(request.content)["stream"] is True
        events = [
            ("message_start", {"message": {"id": "msg1", "type": "message", "role": "assistant", "model": "fixture", "content": [], "usage": {"input_tokens": 10, "output_tokens": 0}}}),
            ("content_block_start", {"index": 0, "content_block": {"type": "thinking", "thinking": ""}}),
            ("content_block_delta", {"index": 0, "delta": {"type": "thinking_delta", "thinking": "分析"}}),
            ("content_block_delta", {"index": 0, "delta": {"type": "signature_delta", "signature": "signature1"}}),
            ("content_block_stop", {"index": 0}),
            ("content_block_start", {"index": 1, "content_block": {"type": "text", "text": ""}}),
            ("content_block_delta", {"index": 1, "delta": {"type": "text_delta", "text": "完成"}}),
            ("content_block_stop", {"index": 1}),
            ("content_block_start", {"index": 2, "content_block": {"type": "tool_use", "id": "tool1", "name": "inspect", "input": {}}}),
            ("content_block_delta", {"index": 2, "delta": {"type": "input_json_delta", "partial_json": '{"x":1}'}}),
            ("content_block_stop", {"index": 2}),
            ("message_delta", {"delta": {"stop_reason": "tool_use", "stop_sequence": None}, "usage": {"output_tokens": 5}}),
            ("message_stop", {}),
        ]
        data = "".join(f"event: {name}\ndata: {json.dumps({'type': name, **value})}\n\n" for name, value in events)
        return httpx.Response(200, request=request, headers={"content-type": "text/event-stream"}, content=data)

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    client = adapter.Model(ModelConfig(
        model="anthropic/fixture", api_key="fixture", base_url="https://ttft.invalid",
        measure_ttft=True, num_retries=0,
    ))
    timings, usages = [], []
    client.request_timing_callback = timings.append
    client.usage_callback = usages.append
    result = asyncio.run(client.completion([{"role": "user", "content": "hello"}]))
    assert result["content"] == "完成"
    assert result["thinking_blocks"][0]["thinking"] == "分析"
    assert result["thinking_blocks"][0]["signature"] == "signature1"
    assert result["tool_calls"][0]["function"] == {"name": "inspect", "arguments": '{"x":1}'}
    assert timings[0]["first_token_kind"] in {"reasoning_content", "thinking"}
    assert usages[0]["prompt_tokens"] == 10
    assert usages[0]["completion_tokens"] == 5
