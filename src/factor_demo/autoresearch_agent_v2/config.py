"""Explicit configuration for the core agent's model access."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr


class TimeBudgetConfig(BaseModel):
    """配置 Router 每轮任务的时间与 token 提醒预算。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seconds: float = Field(default=2400, gt=0, strict=True, allow_inf_nan=False)
    """时间预算秒数；默认 40 分钟，Worker 不独立计时。"""

    tokens: int | None = Field(default=None, gt=0, strict=True)
    """本轮累计 token 提醒预算；None 表示无限。"""


class HistoryConfig(BaseModel):
    """配置上下文预算与自动压缩策略。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    """拒绝未知配置，并保持会话内的压缩策略不变。"""

    context_window_tokens: int = Field(default=172_000, gt=0, strict=True)
    """本会话采用的上下文 token 预算；应按实际模型和服务限制设置。"""

    compact_trigger_percent: float = Field(
        default=80, gt=0, le=100, strict=True, allow_inf_nan=False
    )
    """输入 token 达到上下文预算的该百分比时，在下一次任务请求前压缩。"""

    keep_recent_turns: int = Field(default=2, ge=0, strict=True)
    """压缩时保留的最近交互轮数；一轮包括模型回复及整批工具结果，0 总结全部。"""

    compact_on_worker_reassignment: bool = Field(default=False, strict=True)
    """复用 Worker 执行新任务前，将全部旧任务历史合并为摘要；首次任务跳过。"""


class ModelConfig(BaseModel):
    """Connection and generation settings for a single model."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    """Reject unknown settings and prevent field reassignment."""

    model: str = Field(pattern=r"^(openai|anthropic)/.+$")
    """Provider-prefixed model identifier, such as openai/model-name or anthropic/model-name."""

    api_key: SecretStr = Field(min_length=1, repr=False)
    """Provider credential, masked during configuration serialization and omitted from repr."""

    base_url: str | None = None
    """Custom provider endpoint; None uses the provider's default endpoint."""

    max_tokens: int | None = Field(default=None, gt=0)
    """Generated token cap, including counted thinking tokens; None sets no cap in this layer."""

    num_retries: int = Field(default=10, ge=0)
    """首次请求失败后的最大重试次数；0 关闭重试。"""

    timeout_s: float = Field(default=60, gt=0, allow_inf_nan=False)
    """每次请求的墙钟上限，包含连接和等待完整响应的时间，单位秒。"""

    measure_ttft: bool = Field(default=False, strict=True)
    """通过流式接收测量首个有效输出的等待时间，对外仍返回完整消息。"""

    retry_initial_delay_s: float = Field(default=1, gt=0, allow_inf_nan=False)
    """首次重试前的等待秒数，后续每次翻倍。"""

    retry_max_delay_s: float = Field(default=300, gt=0, allow_inf_nan=False)
    """每次指数退避的最长等待秒数。"""

    thinking: dict[str, Any] | None = None
    """Provider-specific thinking options; None leaves the provider's default behavior."""

    reasoning_effort: str | None = None
    """Provider-supported reasoning effort; None omits this generation option."""

    tool_choice: Literal["auto", "required"] = "required"
    """原生工具选择策略；部分服务在 thinking 模式下只接受 auto。"""
