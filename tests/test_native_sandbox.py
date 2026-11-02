"""Configuration and actual NOOA executor semantics on Linux; no replacement REPL."""

import asyncio
import sys
from dataclasses import replace

import pytest
from nooa.runtime.sandbox.guards import Capabilities
from nooa.strategies import CodeActV2
from nooa.unifiedllm import FakeLLMClient
from test_capabilities import public_case
from test_execution_feedback import cell

from factor_demo.autoresearch_agent import AutoresearchAgent
from factor_demo.capabilities import Capabilities as Policy
from factor_demo.native_sandbox import sandbox_settings
from factor_demo.research_context import install_research_context
from factor_demo.research_workspace import PythonWorkspace
from factor_demo.strategy_factory import research_strategy


def test_profiles_support_no_landlock_without_faking_enforced_guards(monkeypatch):
    caps = Capabilities(linux=True, landlock_abi=0, seccomp=True, rlimit=True)
    monkeypatch.setenv("FACTOR_SANDBOX_FILESYSTEM", "auto")
    monkeypatch.setenv("FACTOR_CELL_ALLOW_NETWORK", "0")
    config = sandbox_settings(capabilities=caps)
    assert not config.filesystem and not config.network and config.require
    monkeypatch.setenv("FACTOR_SANDBOX_FILESYSTEM", "required")
    with pytest.raises(RuntimeError, match="Landlock"):
        sandbox_settings(capabilities=caps)
    monkeypatch.setenv("FACTOR_SANDBOX_FILESYSTEM", "auto")
    with pytest.raises(RuntimeError, match="seccomp"):
        sandbox_settings(capabilities=replace(caps, seccomp=False))
    monkeypatch.setenv("FACTOR_CELL_ALLOW_NETWORK", "1")
    config = sandbox_settings(capabilities=replace(caps, seccomp=False))
    assert config.network and not config.filesystem and config.require
    monkeypatch.setenv("FACTOR_CELL_ALLOW_NETWORK", "oops")
    with pytest.raises(ValueError):
        sandbox_settings(capabilities=caps)


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="native NOOA sandbox is Linux only"
)
@pytest.mark.parametrize("filesystem", ["auto", "off"])
def test_official_strategy_keeps_live_repl_helpers_and_state_after_runtime_error(
    tmp_path, monkeypatch, filesystem
):
    monkeypatch.setenv("FACTOR_SANDBOX_FILESYSTEM", filesystem)

    async def run():
        workspace = PythonWorkspace(public_case(tmp_path / "case"))
        client = FakeLLMClient(
            scripted_responses=[
                cell(
                    "values=[1]\nalias=values\nmarker=id(values)\nx=1\ndef f():\n return x\ng=(i for i in range(3))\nassert next(g)==0",
                    1,
                ),
                cell(
                    'x=3\nassert values is alias and id(values)==marker\nassert f()==3\nassert next(g)==1\nraise ValueError("NATIVE_ERROR_RETAINS_STATE")',
                    2,
                ),
                cell(
                    'assert x==3\nassert f()==3\nassert next(g)==2\nassert callable(methods)\nassert callable(variables)\nassert PredictStrategy is not None\nprint(python_cell_state())\nreturn_result(ResearchReply(summary="native semantics",intent="visualization"))',
                    3,
                ),
            ],
            strict_exhaustion=True,
        )
        agent = AutoresearchAgent(workspace, llm=client)
        install_research_context(agent, "native semantics")
        try:
            strategy = research_strategy(Policy())
            assert type(strategy) is CodeActV2
            assert type(strategy)._create_sandbox_executor is CodeActV2._create_sandbox_executor
            result = await agent.research("probe", _strategy=strategy)
            assert result.summary == "native semantics"
            assert "NATIVE_ERROR_RETAINS_STATE" in str(client.calls[2].messages)
        finally:
            await workspace.close()
            await agent.aclose()

    asyncio.run(run())


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="native sandbox")
def test_network_environment_controls_real_rest_requests(tmp_path, monkeypatch):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"REST_OK")

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("FACTOR_SANDBOX_FILESYSTEM", "off")
    url = f"http://127.0.0.1:{server.server_port}/data"

    async def exercise(allowed):
        monkeypatch.setenv("FACTOR_CELL_ALLOW_NETWORK", str(int(allowed)))
        workspace = PythonWorkspace(public_case(tmp_path / ("on" if allowed else "off")))
        request = f"import httpx\nasync with httpx.AsyncClient(trust_env=False) as c:\n r = await c.get({url!r}, timeout=2)\n text = r.text"
        if allowed:
            code = (
                request
                + "\nassert text=='REST_OK'\nreturn_result(ResearchReply(summary='REST_OK',intent='visualization'))"
            )
        else:
            code = (
                "try:\n "
                + request.replace("\n", "\n ")
                + "\nexcept Exception:\n pass\nelse:\n raise AssertionError('network guard did not block REST')\nreturn_result(ResearchReply(summary='blocked',intent='visualization'))"
            )
        client = FakeLLMClient(scripted_responses=[cell(code, 1)], strict_exhaustion=True)
        agent = AutoresearchAgent(workspace, llm=client)
        install_research_context(agent, "REST policy")
        try:
            result = await agent.research("REST policy", _strategy=research_strategy(Policy()))
            assert result.summary == ("REST_OK" if allowed else "blocked")
        finally:
            await workspace.close()
            await agent.aclose()

    try:
        asyncio.run(exercise(True))
        asyncio.run(exercise(False))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="native sandbox")
def test_native_standalone_predict_helper_can_be_declared_and_called(tmp_path):
    from nooa.unifiedllm import LLMResponse

    class HelperClient(FakeLLMClient):
        async def acall(self, messages, tools=None, output_model=None, **kwargs):
            if output_model is not None:
                parsed = output_model.model_validate({"value": "en"})
                return LLMResponse(content=parsed.model_dump_json(), parsed=parsed)
            return await super().acall(messages, tools=tools, output_model=output_model, **kwargs)

    async def run():
        workspace = PythonWorkspace(public_case(tmp_path / "case"))
        code = '@strategy(PredictStrategy())\nasync def language(text: str) -> str:\n """Return the ISO language code for {text}."""\n ...\ncode=await language("hello")\nassert code=="en"\nreturn_result(ResearchReply(summary=code,intent="visualization"))'
        client = HelperClient(
            scripted_responses=[cell(code, 1)],
            strict_exhaustion=True,
        )
        agent = AutoresearchAgent(workspace, llm=client)
        install_research_context(agent, "native helper")
        try:
            reply = await agent.research("classify language")
            assert reply.summary == "en"
        finally:
            await workspace.close()
            await agent.aclose()

    asyncio.run(run())


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="native sandbox")
def test_native_timeout_restarts_empty_without_replaying_prior_cells(tmp_path):
    async def run():
        workspace = PythonWorkspace(public_case(tmp_path / "case"))
        client = FakeLLMClient(
            scripted_responses=[
                cell("keep=7\nawait asyncio.sleep(10)", 1),
                cell(
                    'try:\n keep\nexcept NameError:\n pass\nelse:\n raise AssertionError("timed-out worker state survived")\nreturn_result(ResearchReply(summary="restarted empty",intent="visualization"))',
                    2,
                ),
            ],
            strict_exhaustion=True,
        )
        agent = AutoresearchAgent(workspace, llm=client)
        install_research_context(agent, "native timeout")
        native = research_strategy(Policy())
        native = CodeActV2(config=native.config.model_copy(update={"cell_timeout": 0.3}))
        try:
            reply = await agent.research("probe", _strategy=native)
            assert reply.summary == "restarted empty"
        finally:
            await workspace.close()
            await agent.aclose()

    asyncio.run(run())
