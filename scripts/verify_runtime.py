"""Small paid GLM check: actual V2 wire settings and native summary application.

Uses the configured reply ceiling with short requested answers. A lower
summary trigger is used only to exercise compression without filling the production input budget.
No credentials, raw requests or reasoning text are written to the report.
"""

from __future__ import annotations

from nooa import Agent, hidden, strategy
from nooa.config import CodeActConfig
from nooa.strategies import CodeActV2

with hidden:
    import asyncio
    import json
    from pathlib import Path
    from urllib.parse import urlparse

    import httpx
    from nooa.context_blocks.events import UserEvent
    from nooa.events import Summary
    from nooa.tracing import enable_tracing, exporters

    from factor_demo.legacy.synthetic import write_json
    from factor_demo.providers import load_provider
    from factor_demo.runtime import (
        RuntimeSettings,
        build_client,
        context_truncation,
        install_summarizer,
    )


class RuntimeProbe(Agent):
    """Read research handoff notes and return short, exact answers."""

    @strategy(CodeActV2(config=CodeActConfig(max_iterations=3, max_retries=1)))
    async def reply(self, request: str) -> str:
        """Follow the request using stored notes. Return the short string via return_result."""
        ...


@hidden
async def main():
    settings = RuntimeSettings.load(Path("configs/runtime.json"))
    provider = load_provider("glm")
    client = build_client(provider, settings)
    host = urlparse(provider.base_url).hostname
    bodies, responses = [], []
    real_send, real_call = httpx.AsyncClient.send, client.acall
    output = Path("runs/runtime-check")
    output.mkdir(parents=True, exist_ok=True)
    enable_tracing(exporters=[exporters.journal_file(output / "traces")])

    async def capture(http_client, request, **kwargs):
        if request.url.host == host and request.method == "POST":
            if len(bodies) >= 6:
                raise RuntimeError("Verification request budget exceeded")
            body = json.loads(request.content)
            safe = {
                k: body[k]
                for k in [
                    "model",
                    "max_tokens",
                    "max_completion_tokens",
                    "reasoning_effort",
                    "thinking",
                    "temperature",
                    "top_p",
                ]
                if k in body
            }
            safe["tools"] = [t.get("function", {}).get("name") for t in body.get("tools", [])]
            assert (
                body.get("max_tokens", body.get("max_completion_tokens"))
                == settings.max_output_tokens
            )
            assert body["reasoning_effort"] == settings.reasoning_effort
            assert body["thinking"]["type"] == "enabled"
            assert safe["tools"] == ["python_cell"]
            bodies.append(safe)
            print("wire", json.dumps(safe), flush=True)
        return await real_send(http_client, request, **kwargs)

    async def call(messages, **params):
        response = await real_call(messages, **params)
        usage = response.usage
        responses.append(
            {
                "finish_reason": response.finish_reason,
                "input_tokens": usage.input_tokens if usage else None,
                "output_tokens": usage.output_tokens if usage else None,
                "reasoning_observed": bool(getattr(response, "reasoning", None)),
            }
        )
        print("response", json.dumps(responses[-1]), flush=True)
        return response

    httpx.AsyncClient.send, client.acall = capture, call
    agent = RuntimeProbe(llm=client, truncation=context_truncation(settings))
    # Lower threshold is a test-only override; production uses the configured budget.
    test_settings = settings.model_copy(
        update={"summary_threshold": 100, "preserve_recent": 1, "summary_target_chars": 1000}
    )
    summarizer = install_summarizer(agent, test_settings)
    facts = ["exp_026", "sensor_037", "recipe_B"]
    notes = UserEvent(
        content="\n".join(
            f"Observation {i}: run-level statistics have been recorded." for i in range(80)
        )
        + "\nHandoff: best experiment exp_026 uses sensor_037 under recipe_B. "
        "Retain these three exact IDs in every summary. These are test identifiers."
    )
    original = agent.event_manager.add(notes)
    try:
        answer = await agent.reply("Acknowledge the notes by returning exactly READY.")
        assert "READY" in answer
        assert summarizer._pending_task is not None, "Native summarizer did not trigger"
        await asyncio.wait_for(summarizer._pending_task, 600)
        summary = summarizer._pending_summary
        assert summary and all(f in summary for f in facts), "Summary failed or omitted handoff IDs"
        summarizer.config = summarizer.config.model_copy(update={"max_tokens": 10**9})
        answer = await agent.reply(
            "Return the best experiment ID, sensor ID and recipe ID from the handoff notes, verbatim."
        )
        assert all(f in answer for f in facts), "Compaction lost a required identifier"
        applied = [e for e in agent.event_manager.values() if isinstance(e, Summary)]
        assert len(applied) == 1, "Background summary was not applied"
        assert original not in list(agent.event_manager.keys())
        assert agent.events[original].id == notes.id, "Archived source event lost"
        result = {
            "verified": True,
            "runtime": settings.public_record(provider),
            "wire_requests": bodies,
            "responses": responses,
            "compression_test": {
                "test_only_trigger": 100,
                "production_trigger": settings.compression_threshold,
                "applied_summaries": len(applied),
                "retained_handoff_ids": facts,
                "raw_source_still_retrievable": True,
            },
            "limitation": f"Short requests accepted with {settings.max_output_tokens} output cap; no full-output or full-context stress test.",
        }
        write_json(Path("docs/runtime-wire-check.json"), result)
        (output / "error.json").unlink(missing_ok=True)
        print(
            "Verified V2, configured output cap, configured effort, and native compaction with fact retention.",
            flush=True,
        )
    except BaseException as error:
        text = str(error).replace(provider.api_key, "[REDACTED]")[:1600]
        write_json(
            output / "error.json",
            {
                "type": type(error).__name__,
                "error": text,
                "wire_requests": bodies,
                "responses": responses,
            },
        )
        print(type(error).__name__, text, flush=True)
        raise
    finally:
        await agent.aclose()
        httpx.AsyncClient.send, client.acall = real_send, real_call
        from opentelemetry import trace

        trace.get_tracer_provider().force_flush()


if __name__ == "__main__":
    asyncio.run(main())
