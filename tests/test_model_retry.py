"""验证请求墙钟、指数退避、重试边界和任务取消，不调用真实模型。"""

import asyncio
from types import SimpleNamespace

import httpx
import litellm
import pytest

from factor_demo.autoresearch_agent_v2.agent import model as adapter
from factor_demo.autoresearch_agent_v2.config import ModelConfig


def model(**options):
    """创建不包含真实凭据的模型适配器。"""
    return adapter.Model(ModelConfig(model="openai/fixture", api_key="fixture", **options))


def record_delays(monkeypatch):
    """仅替换适配器的等待，不影响 SDK 或测试本身的事件循环。"""
    delays = []

    async def sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(adapter, "asyncio", SimpleNamespace(timeout=asyncio.timeout, sleep=sleep))
    return delays


def test_ten_retries_are_capped_and_do_not_mutate_history(monkeypatch):
    """首次请求加十次重试；每次使用独立历史，退避不超过五分钟。"""
    delays = record_delays(monkeypatch)
    calls = []
    error = ConnectionError("fixture")

    async def complete(**kwargs):
        calls.append(kwargs)
        assert kwargs["messages"][0]["content"] == "original"
        kwargs["messages"][0]["content"] = "mutated"
        assert kwargs["timeout"] == 60
        assert kwargs["num_retries"] == kwargs["max_retries"] == 0
        raise error

    monkeypatch.setattr(litellm, "acompletion", complete)
    messages = [{"role": "user", "content": "original"}]
    with pytest.raises(ConnectionError) as caught:
        asyncio.run(model().completion(messages))
    assert caught.value is error
    assert len(calls) == 11
    assert delays == [1, 2, 4, 8, 16, 32, 64, 128, 256, 300]
    assert messages[0]["content"] == "original"


@pytest.mark.parametrize(
    "status,retry",
    [
        (400, False),
        (401, False),
        (403, False),
        (404, False),
        (408, True),
        (409, True),
        (429, True),
        (500, True),
        (503, True),
    ],
)
def test_only_transient_http_errors_retry(monkeypatch, status, retry):
    """暂时故障可恢复，鉴权和参数错误不消耗重试等待。"""
    delays = record_delays(monkeypatch)
    error = litellm.APIError(status, "fixture", "openai", "fixture")
    calls = []
    usages = []

    async def complete(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise error
        return litellm.ModelResponse(
            choices=[{"message": {"role": "assistant", "content": "ok"}}],
            usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        )

    monkeypatch.setattr(litellm, "acompletion", complete)
    client = model()
    client.usage_callback = usages.append
    if retry:
        result = asyncio.run(client.completion([{"role": "user", "content": "hello"}]))
        assert result["content"] == "ok"
        assert len(calls) == 2 and delays == [1]
        assert len(usages) == 1 and usages[0]["total_tokens"] == 12
    else:
        with pytest.raises(litellm.APIError) as caught:
            asyncio.run(client.completion([]))
        assert caught.value is error
        assert len(calls) == 1 and not delays and not usages


def test_stuck_request_hits_wall_clock_and_releases_each_attempt(monkeypatch):
    """即使 SDK 忽略 timeout 参数，适配器仍取消卡住的整次请求。"""
    delays = record_delays(monkeypatch)
    calls = cancelled = 0

    async def complete(**kwargs):
        nonlocal calls, cancelled
        calls += 1
        try:
            await asyncio.Event().wait()
        finally:
            cancelled += 1

    monkeypatch.setattr(litellm, "acompletion", complete)
    with pytest.raises(litellm.Timeout):
        asyncio.run(model(timeout_s=0.01, num_retries=2).completion([]))
    assert calls == cancelled == 3
    assert delays == [1, 2]


@pytest.mark.parametrize("phase", ["request", "backoff"])
def test_task_deadline_interrupts_request_and_backoff_without_retry(monkeypatch, phase):
    """外层截止时间优先，取消不会触发新的请求或等待。"""
    calls = 0

    async def complete(**kwargs):
        nonlocal calls
        calls += 1
        if phase == "backoff":
            raise ConnectionError("fixture")
        await asyncio.Event().wait()

    monkeypatch.setattr(litellm, "acompletion", complete)

    async def run():
        async with asyncio.timeout(0.02):
            await model(retry_initial_delay_s=300).completion([])

    with pytest.raises(TimeoutError):
        asyncio.run(run())
    assert calls == 1


def test_real_sdk_has_no_hidden_retries(monkeypatch):
    """使用真实 SDK 序列化和模拟 HTTP，核对每次适配器尝试只发一个请求。"""
    calls = []
    delays = record_delays(monkeypatch)

    async def send(client, request, **kwargs):
        calls.append(request)
        if len(calls) < 3:
            return httpx.Response(
                503,
                request=request,
                json={
                    "error": {"message": "fixture", "type": "server_error"},
                },
            )
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "fixture",
                "object": "chat.completion",
                "created": 0,
                "model": "fixture",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "ok",
                        },
                    }
                ],
            },
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    client = model(base_url="https://retry.invalid/v1")
    result = asyncio.run(client.completion([{"role": "user", "content": "hello"}]))
    assert result["content"] == "ok"
    assert len(calls) == 3
    assert delays == [1, 2]
