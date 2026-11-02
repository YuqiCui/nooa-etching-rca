"""Asynchronous model completions preserving complete assistant messages."""

from collections.abc import Callable
from copy import deepcopy
from typing import Any

import litellm

from ..config import ModelConfig


class Model:
    """Model access through LiteLLM's unified message format."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        usage_callback: Callable[[dict[str, Any] | None], None] | None = None,
    ):
        """配置模型和可选用量回调，不发送请求。

        Args:
            config: 模型服务、凭据和生成设置。
            usage_callback: 每次收到响应时同步接收原始用量字典；服务未报告时接收 None。
                回调异常向上传播，不影响默认的消息返回结构。
        """
        self.config = config
        self.usage_callback = usage_callback

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
    ) -> dict[str, Any]:
        """调用模型并返回完整的 assistant 消息。

        Args:
            messages: LiteLLM 统一格式的完整有序历史，包含推理、签名和工具调用。
            tools: 可调用工具的 JSON Schema；选择策略由 config.tool_choice 指定。

        Returns:
            LiteLLM 返回的完整消息字典，保留推理、thinking 块、工具调用和扩展字段。

        Raises:
            Exception: 模型服务和传输错误原样向上传播。
            ValueError: 模型响应没有消息选项。
        """
        options: dict[str, Any] = {}
        if tools is not None:
            options["tools"] = deepcopy(tools)
            options["tool_choice"] = self.config.tool_choice
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
        response = await litellm.acompletion(
            model=self.config.model,
            messages=deepcopy(messages),
            api_key=self.config.api_key.get_secret_value(),
            api_base=self.config.base_url,
            timeout=self.config.timeout_s,
            stream=False,
            n=1,
            num_retries=self.config.num_retries,
            drop_params=False,
            **options,
        )
        if self.usage_callback is not None:
            # LiteLLM 在服务未报告用量时可能完全省略 usage 属性。
            try:
                usage = response.usage
            except AttributeError:
                usage = None
            self.usage_callback(deepcopy(usage.model_dump()) if usage is not None else None)
        if not response.choices:
            raise ValueError("Model response contains no message choice")
        return response.choices[0].message.model_dump()
