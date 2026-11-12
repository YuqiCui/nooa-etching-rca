"""验证启动清单、实际请求提示词和延迟实例化边界。"""

import asyncio
from contextlib import asynccontextmanager

import pytest

from factor_demo.autoresearch_agent_v2.agent import Session, SessionState
from factor_demo.autoresearch_agent_v2.agent.model import Model
from factor_demo.autoresearch_agent_v2.config import ModelConfig
from factor_demo.autoresearch_agent_v2.python_tools import PythonTool, PythonTools
from factor_demo.autoresearch_agent_v2.tools.registry import ToolRegistry


def make_session(tmp_path, **options):
    """创建占位模型会话，不发送真实模型请求。"""
    state = SessionState(
        session_id="startup-test", agent_id="agent", task_id="task", workdir=tmp_path
    )
    model = Model(ModelConfig(model="openai/fixture", api_key="private-fixture-key"))
    return Session(model, state, **options)


def test_startup_prints_effective_prompt_and_bound_catalog(tmp_path, monkeypatch, capsys):
    """提示词与实际请求相同，清单来自已绑定会话，输出不创建业务工具。"""
    monkeypatch.setattr(PythonTool, "_registry", {})

    @PythonTool.register("LazyTool")
    class LazyTool(PythonTool):
        """测试用延迟创建工具。"""

        def __init__(self):
            """若启动日志触发业务工具实例化则立即失败。"""
            raise AssertionError("业务工具不应在启动时创建")

        def methods(self):
            """满足抽象接口，启动日志不应读取方法清单。"""
            raise AssertionError("启动时不应读取方法清单")

    requests = []

    async def completion(self, messages, *, tools=None):
        requests.append(messages)
        return {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "finish-call",
                    "type": "function",
                    "function": {"name": "finish", "arguments": '{"content":"ok"}'},
                }
            ],
        }

    monkeypatch.setattr(Model, "completion", completion)
    registry = ToolRegistry()
    registry.register_python("LazyTool", LazyTool, methods=("read",))
    session = make_session(tmp_path, system_prompt="自定义系统提示", tool_registry=registry)
    session.register_dataset("样本", "data/sample.parquet", "逐片观测")
    session.register_stateful_tool(PythonTools)
    session.loop.tools.add_instructions("额外工具说明")
    assert capsys.readouterr().err == ""

    async def scenario():
        async with session:
            output = capsys.readouterr().err
            assert "[Session started] startup-test" in output
            logged_prompt = output.split("=== System prompt ===\n", 1)[1].split(
                "\n\n=== Datasets (", 1
            )[0]
            assert logged_prompt.startswith("自定义系统提示\n\n")
            assert "额外工具说明" in logged_prompt
            assert "# 工具调用输出协议" not in logged_prompt
            assert "=== Datasets (1) ===" in output
            assert '"name": "样本"' in output
            assert '"path": "data/sample.parquet"' in output
            assert '"description": "逐片观测"' in output
            for schema in session.schemas():
                assert f"- {schema['function']['name']}:" in output
            assert "=== Python tools (1) ===\n- LazyTool" in output
            assert "private-fixture-key" not in output
            assert session.state.messages == []
            await session.start()
            assert capsys.readouterr().err == ""
            assert await session.run("继续") == "ok"
            assert logged_prompt == requests[0][0]["content"]
            assert "[Session started]" not in capsys.readouterr().err

    asyncio.run(scenario())


def test_implicit_start_prints_empty_catalogs_once(tmp_path, monkeypatch, capsys):
    """首次 run 自动打印，未绑定 PythonTools 时不列出全局注册类。"""

    async def completion(self, messages, *, tools=None):
        return {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "finish-call",
                    "type": "function",
                    "function": {"name": "finish", "arguments": '{"content":"ok"}'},
                }
            ],
        }

    monkeypatch.setattr(Model, "completion", completion)
    session = make_session(tmp_path)

    async def scenario():
        try:
            await session.run("开始")
            await session.run("继续")
        finally:
            await session.aclose()

    asyncio.run(scenario())
    captured = capsys.readouterr()
    output = captured.err
    assert captured.out == "ok\nok\n"
    assert output.count("[Session started]") == 1
    assert "=== Datasets (0) ===\n(none)" in output
    assert "=== Tools (2) ===" in output
    assert "=== Python tools (0) ===\n(none)" in output


def test_failed_startup_does_not_print_success(tmp_path, capsys):
    """资源绑定失败时不打印成功清单。"""

    @asynccontextmanager
    async def failing_resource(session):
        """模拟资源工厂在返回实例前失败。"""
        raise ValueError("startup failed")
        yield object()

    session = make_session(tmp_path)
    session.register_resource(object, failing_resource)
    with pytest.raises(ValueError, match="startup failed"):
        asyncio.run(session.start())
    assert capsys.readouterr().err == ""
