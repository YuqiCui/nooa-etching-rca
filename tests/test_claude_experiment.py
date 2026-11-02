"""Regression checks for the experimental CLI and business-tool boundary."""

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from litellm import ModelResponse
from mcp.server.fastmcp.exceptions import ToolError
from test_capabilities import public_case

from factor_demo.claude_experiment.__main__ import _prepare_run
from factor_demo.claude_experiment.gateway import create_gateway
from factor_demo.claude_experiment.report import _claude_metrics
from factor_demo.claude_experiment.tools import clean_environment, create_server
from factor_demo.providers import Provider
from factor_demo.runtime import RuntimeSettings


def test_credential_environment_is_removed():
    assert clean_environment(
        {
            "PATH": "/usr/bin",
            "FACTOR_API_KEY": "secret",
            "ANTHROPIC_API_KEY": "secret",
            "GITHUB_TOKEN": "secret",
            "AWS_SECRET_ACCESS_KEY": "secret",
            "CLAUDECODE": "1",
        }
    ) == {"PATH": "/usr/bin"}


def test_report_includes_timeout_without_execution_reference(tmp_path):
    formal = tmp_path / "formal"
    formal.mkdir()
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "model": "deepseek-v4-flash",
                "status": "running",
            }
        )
    )
    rows = [
        {
            "tool": "run_code",
            "ok": False,
            "elapsed_s": 120.5,
            "result": {"ok": False, "error": "timeout"},
        }
    ]
    (formal / "tool-calls.jsonl").write_text("\n".join(map(json.dumps, rows)))
    result = _claude_metrics(tmp_path)
    assert result["scientific_executions"] == 1
    assert result["failed_scientific_executions"] == 1
    assert result["scientific_elapsed_s"] == 120.5


def test_public_copy_excludes_outputs_and_detects_changed_inputs(tmp_path):
    source = public_case(tmp_path / "baseline")
    (source / "workspace/private-result.txt").write_text("previous finding")
    assert _prepare_run(source, tmp_path / "new") == 2
    assert not (tmp_path / "new/workspace/private-result.txt").exists()
    (source / "input/recipe.json").write_text("tampered")
    with pytest.raises(ValueError, match="baseline hashes"):
        _prepare_run(source, tmp_path / "tampered")


def test_mcp_exposes_typed_business_tools_and_journals_failure(tmp_path):
    async def check():
        run = public_case(tmp_path / "case")
        server = create_server(run)
        tools = {tool.name: tool for tool in await server.list_tools()}
        assert "run_code" in tools and "inspect_dataset" in tools
        assert "research" not in tools and "spawn_research" not in tools
        assert "Bash" not in tools and "Read" not in tools
        assert tools["run_code"].inputSchema["required"] == ["code"]
        with pytest.raises(ToolError, match="artifacts"):
            await server.call_tool("read_artifact", {"path": "../.env"})
        record = json.loads((run / "tool-calls.jsonl").read_text())
        assert record["ok"] is False and record["elapsed_s"] >= 0

    asyncio.run(check())


def test_gateway_pins_upstream_settings_and_serializes_streams(tmp_path, monkeypatch):
    async def check():
        requests = []
        active, maximum = 0, 0

        async def fake_create(**body):
            nonlocal active, maximum
            requests.append(body)
            active += 1
            maximum = max(maximum, active)

            async def events():
                nonlocal active
                await asyncio.sleep(0.01)
                yield ModelResponse(
                    stream=True,
                    choices=[
                        {
                            "index": 0,
                            "delta": {"content": "OK"},
                            "finish_reason": None,
                        }
                    ],
                )
                yield ModelResponse(
                    stream=True,
                    choices=[
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                )
                active -= 1

            return events()

        monkeypatch.setattr("litellm.acompletion", fake_create)
        provider = Provider(
            "custom",
            "deepseek-v4-flash",
            "https://example.com/v1",
            "openai",
            "private-upstream-key",
        )
        app = create_gateway(provider, RuntimeSettings(), "local-key", tmp_path, max_calls=2)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as c:
            payload = {
                "model": provider.model,
                "max_tokens": 1,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            }
            assert (await c.post("/v1/messages", json=payload)).status_code == 401
            headers = {"x-api-key": "local-key"}
            results = await asyncio.gather(
                *(c.post("/v1/messages", json=payload, headers=headers) for _ in range(2))
            )
            assert all(r.status_code == 200 for r in results)
            assert maximum == 1
            assert (await c.post("/v1/messages", json=payload, headers=headers)).status_code == 429
        assert requests[0]["model"] == "openai/deepseek-v4-flash"
        assert requests[0]["api_base"] == provider.base_url
        assert requests[0]["api_key"] == provider.api_key
        assert requests[0]["reasoning_effort"] == "max"
        assert requests[0]["max_tokens"] == 50000
        assert requests[0]["extra_body"] == {"thinking": {"type": "enabled"}}
        for path in Path(tmp_path).glob("*.jsonl"):
            assert provider.api_key not in path.read_text()

    asyncio.run(check())
