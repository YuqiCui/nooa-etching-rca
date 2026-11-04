"""Explicit model budgets and provider-native reasoning controls."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from .providers import Provider


class RuntimeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    context_window: int = 272_000
    max_output_tokens: int = 50_000
    thinking: bool = True
    reasoning_effort: Literal["low", "medium", "high", "max"] = "medium"
    anthropic_thinking_budget: int = 32_000
    summary_threshold: int | None = None
    preserve_recent: int = 10
    summary_target_chars: int = 8000
    request_timeout: int = 1800
    review_provider: str = "auto"

    @model_validator(mode="after")
    def validate_limits(self):
        if not 0 < self.max_output_tokens < self.context_window:
            raise ValueError("Output cap must be positive and smaller than total context")
        if (
            self.summary_threshold is not None
            and not 0 < self.summary_threshold < self.input_budget
        ):
            raise ValueError("Summary threshold must be below the usable input budget")
        if self.preserve_recent < 1 or self.summary_target_chars < 100:
            raise ValueError("Invalid summary settings")
        if not 0 < self.anthropic_thinking_budget < self.max_output_tokens:
            raise ValueError("Thinking budget must leave room for an answer")
        if self.request_timeout < 1:
            raise ValueError("Invalid request timeout")
        return self

    @property
    def input_budget(self) -> int:
        return self.context_window - self.max_output_tokens

    @property
    def compression_threshold(self) -> int:
        return self.summary_threshold or int(self.input_budget * 0.8)

    @classmethod
    def load(cls, path: Path) -> RuntimeSettings:
        return cls.model_validate(json.loads(path.read_text()))

    def public_record(self, provider: Provider) -> dict:
        record = self.model_dump()
        record.update(
            strategy="CodeActV2",
            usable_input_tokens=self.input_budget,
            max_context_tokens=self.input_budget,
            effective_summary_threshold=self.compression_threshold,
            summarizer="NOOA TokenBudgetSummarizer",
            token_unit="decimal tokens",
        )
        if provider.api_style == "anthropic":
            record["reasoning_effort"] = None
            record["reasoning_control"] = "thinking.budget_tokens (not a native effort field)"
        else:
            record["reasoning_control"] = "reasoning_effort"
        return record


def build_client(provider: Provider, settings: RuntimeSettings):
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    from nooa.unifiedllm import HttpConfig, RetryConfig
    from nooa.unifiedllm.registry import client_from_config

    kwargs = {
        "api_base": provider.base_url,
        "api_key": provider.api_key,
        "max_tokens": settings.max_output_tokens,
        "drop_params": False,
        "retry_config": RetryConfig(max_retries=1, rate_limit_extra_retries=0),
        "http_config": HttpConfig(
            read_timeout=settings.request_timeout, request_timeout=settings.request_timeout
        ),
    }
    if provider.api_style == "anthropic":
        prefix = "anthropic/"
        kwargs["allowed_openai_params"] = ["thinking"]
        kwargs["thinking"] = (
            {"type": "enabled", "budget_tokens": settings.anthropic_thinking_budget}
            if settings.thinking
            else {"type": "disabled"}
        )
    else:
        prefix = "openai/"
        if provider.model.lower().startswith("glm-5.3") and not settings.thinking:
            raise ValueError("GLM-5.3 cannot disable thinking; select effort low instead")
        kwargs["reasoning_effort"] = settings.reasoning_effort
        kwargs["allowed_openai_params"] = ["reasoning_effort"]
        kwargs["extra_body"] = {
            "thinking": {"type": "enabled" if settings.thinking else "disabled"}
        }
        if provider.model.lower().startswith("glm-"):
            kwargs["extra_body"]["thinking"]["clear_thinking"] = False
            kwargs.update(temperature=1.0, top_p=0.95)
    # Capacity is registry metadata, not an endpoint generation parameter.
    client = client_from_config(
        "factor-demo-" + provider.name,
        {"model_name": prefix + provider.model, "context_window": settings.context_window},
        **kwargs,
    )

    if provider.api_style == "anthropic" and provider.model.lower().startswith("minimax"):
        from .provider_compat import install_json_transport

        install_json_transport(client)
    return client


def install_summarizer(agent, settings: RuntimeSettings):
    from nooa.agents import TokenBudgetSummarizer
    from nooa.config.summarizer_config import TokenBudgetConfig

    return TokenBudgetSummarizer.install(
        agent,
        config=TokenBudgetConfig(
            max_tokens=settings.compression_threshold,
            preserve_recent=settings.preserve_recent,
            target_chars=settings.summary_target_chars,
        ),
    )


def context_truncation(settings: RuntimeSettings):
    from nooa.config.truncation_config import TruncationConfig

    return TruncationConfig(max_context_tokens=settings.input_budget)
