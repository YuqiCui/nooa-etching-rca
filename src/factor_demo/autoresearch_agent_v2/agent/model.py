"""Asynchronous model completions preserving complete assistant messages."""

import asyncio
import logging
import time
import uuid
from asyncio import CancelledError
from collections.abc import Callable
from contextlib import suppress
from contextvars import ContextVar
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any, Literal

import httpx
import litellm

from ..config import ModelConfig

logger = logging.getLogger(__name__)


def _first_output_kind(chunk) -> str | None:
    """识别首个有效增量，跳过空包、角色声明和纯用量包。"""
    for choice in chunk.choices:
        delta = choice.delta.model_dump()
        for field in ("reasoning_content", "content"):
            if delta.get(field):
                return field
        if any(block.get("thinking") for block in delta.get("thinking_blocks") or []):
            return "thinking"
        for call in delta.get("tool_calls") or []:
            function = call.get("function") or {}
            if function.get("name") or function.get("arguments"):
                return "tool_calls"
    return None


def _is_retryable(error: Exception) -> bool:
    """只重试暂时的传输、限流和服务故障，配置与鉴权错误直接返回。"""
    if isinstance(error, (
        TimeoutError, ConnectionError, httpx.TransportError,
        litellm.Timeout, litellm.APIConnectionError,
    )):
        return True
    status = getattr(error, "status_code", None)
    return isinstance(status, int) and (status in {408, 409, 429} or 500 <= status < 600)


class Model:
    """Model access through LiteLLM's unified message format."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        usage_callback: Callable[[dict[str, Any] | None], None] | None = None,
        request_timing_callback: Callable[[dict[str, Any]], None] | None = None,
    ):
        """配置模型和可选用量回调，不发送请求。

        Args:
            config: 模型服务、凭据和生成设置。
            usage_callback: 每次收到响应时同步接收原始用量字典；服务未报告时接收 None。
                回调异常向上传播，不影响默认的消息返回结构。
            request_timing_callback: 接收每次请求尝试的首输出和终态计时事件。
        """
        self.config = config
        self.usage_callback = usage_callback
        self.request_timing_callback = request_timing_callback
        self._reported_tokens: ContextVar[int] = ContextVar("reported_tokens", default=0)

    @property
    def reported_tokens(self) -> int:
        """返回当前异步执行上下文中累计收到的模型用量。"""
        return self._reported_tokens.get()

    def _record_timing(self, record: dict[str, Any]) -> None:
        """发布计时事件，统计保存失败不重试模型请求。"""
        if self.request_timing_callback is not None:
            try:
                self.request_timing_callback(deepcopy(record))
            except Exception:
                logger.exception("保存 API 请求计时失败")

    async def _collect_stream(self, stream, messages, timing, started):
        """统计首输出并用 SDK 合并完整消息，退出时释放连接。"""
        chunks = []
        thinking_text = ""
        try:
            async for chunk in stream:
                # SDK 在结束时也会合并并修改原包；保存快照避免重复拼接思考文本。
                snapshot = deepcopy(chunk)
                if self.config.model.startswith("anthropic/"):
                    for choice in snapshot.choices:
                        for block in choice.delta.get("thinking_blocks") or []:
                            text = block.get("thinking", "")
                            # 当前 SDK 的签名包可能附带已发送的整段思考，不能再次累加。
                            if block.get("signature") and text == thinking_text:
                                block["thinking"] = ""
                            else:
                                thinking_text += text
                            if block.get("signature"):
                                thinking_text = ""
                chunks.append(snapshot)
                if timing["ttft_s"] is None and (kind := _first_output_kind(chunk)):
                    timing.update(ttft_s=time.perf_counter() - started, first_token_kind=kind)
                    self._record_timing({**timing, "event": "first_token"})
            # LiteLLM 会另行生成估算用量包；只从其原始包判定服务端是否报告用量。
            source_chunks = getattr(stream, "chunks", chunks)
            reported_usage = any(getattr(chunk, "usage", None) is not None
                                 for chunk in source_chunks)
            response = litellm.stream_chunk_builder(chunks, messages=messages)
            if response is None:
                raise ValueError("模型流未返回可合并的消息")
            if not reported_usage:
                # SDK 可能估算用量；原接口只接受服务端实际报告的用量。
                response.usage = None
            return response
        finally:
            with suppress(Exception):
                await stream.aclose()

    def count_tokens(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> int:
        """估算下一次请求的输入 token 数，不发送模型请求或读取远程图片。

        Args:
            messages: 实际准备发送的消息，含系统说明、推理、调用与工具结果。
            tools: 原生调用模式使用的工具定义；JSON 模式已将定义写入系统消息。

        Returns:
            LiteLLM 根据模型 tokenizer 估算的输入 token 数；图片使用默认预算。
            未知模型使用库的默认 tokenizer，结果不等同于服务端账单用量。

        Raises:
            ValueError: 计数器关闭或未返回正的整数。
            Exception: tokenizer 或消息格式错误原样向上传播。
        """
        count = litellm.token_counter(
            model=self.config.model,
            messages=deepcopy(messages),
            tools=deepcopy(tools),
            use_default_image_token_count=True,
        )
        if type(count) is not int or count <= 0:
            raise ValueError("输入 token 计数不可用，请检查 LiteLLM 计数器配置")
        return count

    async def completion(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Literal["auto", "required", "none"] | None = None,
    ) -> dict[str, Any]:
        """调用模型并返回完整的 assistant 消息。

        Args:
            messages: LiteLLM 统一格式的完整有序历史，包含推理、签名和工具调用。
            tools: 可调用工具的 JSON Schema。
            tool_choice: 本次覆盖工具选择策略；None 沿用 config.tool_choice。

        Returns:
            LiteLLM 返回的完整消息字典，保留推理、thinking 块、工具调用和扩展字段。

        Raises:
            Exception: 模型服务和传输错误原样向上传播。
            ValueError: 模型响应没有消息选项。
        """
        options: dict[str, Any] = {}
        if tools is not None:
            options["tools"] = deepcopy(tools)
            options["tool_choice"] = tool_choice or self.config.tool_choice
        if self.config.max_tokens is not None:
            options["max_tokens"] = self.config.max_tokens
        if self.config.thinking is not None:
            if self.config.model.startswith("openai/"):
                # 兼容服务的扩展字段必须放入请求体，不能作为 OpenAI SDK 的命名参数。
                options["extra_body"] = {"thinking": deepcopy(self.config.thinking)}
            else:
                options["thinking"] = deepcopy(self.config.thinking)
                options["allowed_openai_params"] = ["thinking"]
        if self.config.reasoning_effort is not None:
            options["reasoning_effort"] = self.config.reasoning_effort
        response = await self._request_with_retries(messages, options)
        # LiteLLM 在服务未报告用量时可能完全省略 usage 属性。
        usage = getattr(response, "usage", None)
        if usage is not None:
            total = usage.total_tokens
            if total is None:
                prompt = usage.prompt_tokens
                completion = usage.completion_tokens
                if prompt is not None and completion is not None:
                    total = prompt + completion
            if total is not None and total > 0:
                self._reported_tokens.set(self.reported_tokens + total)
        if self.usage_callback is not None:
            self.usage_callback(deepcopy(usage.model_dump()) if usage is not None else None)
        if not response.choices:
            raise ValueError("Model response contains no message choice")
        return response.choices[0].message.model_dump()

    async def _request_with_retries(self, messages, options):
        """逐次限制整次请求时间；总任务取消可立即打断请求或退避。"""
        delay = min(self.config.retry_initial_delay_s, self.config.retry_max_delay_s)
        for attempt in range(self.config.num_retries + 1):
            try:
                return await self._request_once(messages, options)
            except Exception as error:
                if attempt == self.config.num_retries or not _is_retryable(error):
                    raise
                logger.warning(
                    "模型请求失败（%s），%.3g 秒后重试（%d/%d）",
                    type(error).__name__, delay, attempt + 1, self.config.num_retries,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, self.config.retry_max_delay_s)

    async def _request_once(self, messages, options):
        """关闭隐式重试，并将单次请求超时与研究总时限区分开。"""
        timing = {
            "attempt_id": uuid.uuid4().hex,
            "started_at": datetime.now(UTC).isoformat(),
            "stream": self.config.measure_ttft,
            "ttft_s": None,
            "first_token_kind": None,
        }
        started = time.perf_counter()
        status = "failed"
        error_type = None
        try:
            async with asyncio.timeout(self.config.timeout_s):
                response = await litellm.acompletion(
                    model=self.config.model,
                    messages=deepcopy(messages),
                    api_key=self.config.api_key.get_secret_value(),
                    api_base=self.config.base_url,
                    timeout=self.config.timeout_s,
                    stream=self.config.measure_ttft,
                    **({"stream_options": {"include_usage": True}}
                       if self.config.measure_ttft else {}),
                    n=1,
                    num_retries=0,
                    max_retries=0,
                    drop_params=False,
                    **deepcopy(options),
                )
                if self.config.measure_ttft:
                    response = await self._collect_stream(response, messages, timing, started)
                status = "completed"
                return response
        except TimeoutError as error:
            error_type = "Timeout"
            raise litellm.Timeout(
                message=f"模型请求超时，单次上限为 {self.config.timeout_s:g} 秒",
                model=self.config.model,
                llm_provider=self.config.model.split("/", 1)[0],
            ) from error
        except BaseException as error:
            error_type = type(error).__name__
            if isinstance(error, CancelledError):
                status = "cancelled"
            raise
        finally:
            self._record_timing({
                **timing, "event": "end", "status": status,
                "duration_s": time.perf_counter() - started, "error_type": error_type,
            })
