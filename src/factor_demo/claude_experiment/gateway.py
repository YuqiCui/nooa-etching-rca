"""Translate Claude's Messages protocol to the unchanged OpenAI upstream."""

import asyncio
import json
import time
from pathlib import Path

import litellm
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.anthropic.pass_through.adapters.transformation import AnthropicAdapter

from factor_demo.providers import Provider
from factor_demo.runtime import RuntimeSettings


def append_record(path: Path, record: dict) -> None:
    """Append one JSON audit record without credential fields.

    Args:
        path: Experiment-owned journal.
        record: Public request metadata, response, or usage record.
    """
    with path.open("a") as output:
        output.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


class UsageJournal(CustomLogger):
    """Keep upstream token counts before Anthropic protocol conversion."""

    def __init__(self, path: Path):
        super().__init__()
        self.path = path
        self.total_tokens = 0

    def log_pre_api_call(self, model, messages, kwargs):
        """Record resolved upstream generation fields, excluding credentials.

        Args:
            model: Resolved upstream model name.
            messages: Converted OpenAI conversation.
            kwargs: SDK metadata containing resolved generation parameters.
        """
        params = {
            **kwargs.get("optional_params", {}),
            **kwargs.get("additional_args", {}).get("complete_input_dict", {}),
        }
        append_record(
            self.path.with_name("upstream-parameters.jsonl"),
            {
                "time": time.time(),
                "model": model,
                "parameters": {
                    name: params[name]
                    for name in (
                        "model",
                        "max_tokens",
                        "max_completion_tokens",
                        "reasoning_effort",
                        "thinking",
                        "stream",
                        "stream_options",
                    )
                    if name in params
                },
            },
        )

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Record raw usage from one completed upstream request.

        Args:
            kwargs: LiteLLM callback metadata.
            response_obj: Complete upstream response, including streaming usage.
            start_time: SDK request start time.
            end_time: SDK request completion time.
        """
        raw_usage = response_obj.usage
        usage = raw_usage if isinstance(raw_usage, dict) else raw_usage.model_dump()
        self.total_tokens += usage.get("total_tokens", 0)
        append_record(self.path, {"time": time.time(), "usage": usage})


def create_gateway(
    provider: Provider, settings: RuntimeSettings, key: str, run: Path, max_calls: int = 256
) -> FastAPI:
    """Create an authenticated, serial gateway pinned to the original model.

    Args:
        provider: Original upstream endpoint, model and secret.
        settings: Original token, reasoning and request limits.
        key: Ephemeral loopback credential, never sent upstream.
        run: Directory for request and usage journals.
        max_calls: Maximum upstream model requests, including retries at the CLI layer.

    Returns:
        A FastAPI app serving the Anthropic Messages endpoint.
    """
    litellm.use_chat_completions_url_for_anthropic_messages = True
    journal = UsageJournal(run / "upstream-usage.jsonl")
    litellm.callbacks = [journal]
    gate = asyncio.Semaphore(1)
    calls = 0
    app = FastAPI()

    @app.post("/v1/messages")
    async def messages(request: Request):
        nonlocal calls
        if request.headers.get("x-api-key") != key:
            raise HTTPException(401, "Invalid experiment gateway key")
        body = await request.json()
        if body.get("model") != provider.model:
            raise HTTPException(404, "Only the original experiment model is permitted")
        await gate.acquire()
        try:
            if calls >= max_calls or journal.total_tokens >= 4_000_000:
                raise HTTPException(429, "Experiment budget exhausted")
            calls += 1
            started = time.time()
            append_record(
                run / "gateway-requests.jsonl",
                {
                    "call": calls,
                    "started": started,
                    "model": provider.model,
                    "max_tokens": settings.max_output_tokens,
                    "reasoning_effort": settings.reasoning_effort,
                    "thinking": settings.thinking,
                    "request": body,
                },
            )
            # Use the conversion primitives, then the same completion API as NOOA.
            # The SDK's Messages launcher normalizes max effort and reroutes models.
            adapter = AnthropicAdapter()
            translated, tool_names = adapter.translate_completion_input_params_with_tool_mapping(
                {
                    name: body[name]
                    for name in (
                        "model",
                        "messages",
                        "system",
                        "tools",
                        "tool_choice",
                        "max_tokens",
                        "stop_sequences",
                        "temperature",
                        "top_p",
                        "stream",
                    )
                    if name in body
                },
                custom_llm_provider="openai",
            )
            translated.update(
                model="openai/" + provider.model,
                api_base=provider.base_url,
                api_key=provider.api_key,
                max_tokens=settings.max_output_tokens,
                reasoning_effort=settings.reasoning_effort,
                extra_body={"thinking": {"type": "enabled" if settings.thinking else "disabled"}},
                timeout=settings.request_timeout,
                num_retries=1,
                drop_params=False,
            )
            if body.get("stream"):
                translated["stream_options"] = {"include_usage": True}
            try:
                response = await litellm.acompletion(**translated)
            except (
                litellm.APIError,
                litellm.AuthenticationError,
                litellm.BadRequestError,
            ) as error:
                message = str(error).replace(provider.api_key, "[REDACTED]")
                append_record(
                    run / "gateway-errors.jsonl",
                    {
                        "time": time.time(),
                        "type": type(error).__name__,
                        "message": message,
                    },
                )
                gate.release()
                return JSONResponse(
                    status_code=502,
                    content={
                        "type": "error",
                        "error": {"type": "api_error", "message": message},
                    },
                )
            if not body.get("stream"):
                gate.release()
                return adapter.translate_completion_output_params(response, tool_names)
            response = adapter.translate_completion_output_params_streaming(
                response,
                model=provider.model,
                tool_name_mapping=tool_names,
            )
        except BaseException:
            gate.release()
            raise

        async def events():
            # A streaming request keeps the permit until its final event.
            try:
                async for chunk in response:
                    if isinstance(chunk, (bytes, str)):
                        text = chunk.decode() if isinstance(chunk, bytes) else chunk
                        append_record(
                            run / "gateway-events.jsonl",
                            {
                                "time": time.time(),
                                "sse": text,
                            },
                        )
                        yield chunk
                        continue
                    event = (
                        chunk if isinstance(chunk, dict) else chunk.model_dump(exclude_none=True)
                    )
                    append_record(run / "gateway-events.jsonl", {"time": time.time(), **event})
                    yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
            finally:
                gate.release()

        return StreamingResponse(events(), media_type="text/event-stream")

    return app
