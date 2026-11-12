"""检查实例工具目录、延迟绑定和全部调用入口的授权约束。"""

import asyncio
from contextlib import asynccontextmanager

import pytest

from factor_demo.autoresearch_agent_v2.agent import Session, SessionState
from factor_demo.autoresearch_agent_v2.agent.model import Model
from factor_demo.autoresearch_agent_v2.config import ModelConfig
from factor_demo.autoresearch_agent_v2.python_tools import PythonTool, PythonTools
from factor_demo.autoresearch_agent_v2.tools.base import ToolFactory
from factor_demo.autoresearch_agent_v2.tools.policy import ToolPolicy
from factor_demo.autoresearch_agent_v2.tools.registry import ToolRegistry
from factor_demo.business_tools import register_business_tools


def make_session(tmp_path, registry, allow_tools="all"):
    """仅创建占位模型与工具资源，不调用真实模型。"""
    state = SessionState(session_id="test", agent_id="main", task_id="test", workdir=tmp_path)
    return Session(
        Model(ModelConfig(model="openai/fixture", api_key="fixture")),
        state,
        tool_registry=registry,
        allow_tools=allow_tools,
    )


def test_registry_instances_and_copies_keep_registration_local():
    first, second = ToolRegistry(), ToolRegistry()
    register_business_tools(first)
    copied = first.copy()
    first.register_callable("extra", lambda: 1, description="附加工具。")
    assert second.names() == ()
    assert "extra" not in copied.names()
    assert copied.providers()["business:Statistics"] == (
        "Statistics.summarize",
        "Statistics.pearson_correlation",
    )


def test_explicit_policy_expands_provider_and_rejects_unknown_names():
    policy = ToolPolicy(
        ["finish", "business:Statistics"],
        ["finish", "Statistics.summarize"],
        {"business:Statistics": ["Statistics.summarize"]},
    )
    assert policy.allowed_names == {"finish", "Statistics.summarize"}
    with pytest.raises(ValueError, match="未知"):
        ToolPolicy(["missing"], ["finish"])
    with pytest.raises(TypeError, match="allow_tools"):
        ToolPolicy("finish", ["finish"])


def test_factory_filters_schema_instructions_and_real_calls():
    factory = ToolFactory()
    factory.register_callable("read", lambda: 1, description="读取。")
    factory.register_callable("write", lambda: 2, description="写入。")
    with factory.registration_scope(("read", "write")):
        factory.add_instructions("使用 read 和 write。")
    factory.add_instructions("只读说明。", names=("read",))
    factory.restrict(ToolPolicy(["read"], factory.names()))
    assert [item["function"]["name"] for item in factory.schemas()] == ["read"]
    assert factory.instructions() == "只读说明。"
    assert asyncio.run(factory.call("read", {})) == 1
    with pytest.raises(PermissionError):
        asyncio.run(factory.call("write", {}))


def test_denied_stateful_group_never_binds_resources(tmp_path):
    class UnusedTools:
        """若授权检查太晚就会导致启动失败的工具组。"""

        namespace = "unused"
        tool_names = ("unused__run",)

        @classmethod
        @asynccontextmanager
        async def bind(cls, session):
            """被拒绝的组不得进入绑定工厂。"""
            raise AssertionError("未授权资源被创建")
            yield

    registry = ToolRegistry()
    registry.register_stateful(UnusedTools)

    async def scenario():
        async with make_session(tmp_path, registry, ["finish"]) as session:
            assert [item["function"]["name"] for item in session.schemas()] == ["finish"]
            with pytest.raises(KeyError):
                await session.call("unused__run", {})

    asyncio.run(scenario())


def test_session_rejects_unknown_selection_and_missing_finish_before_binding(tmp_path):
    for selection in (["finish", "missing"], []):

        async def scenario(selected=selection):
            with pytest.raises(ValueError):
                async with make_session(tmp_path, ToolRegistry(), selected):
                    pass

        asyncio.run(scenario())


def test_python_method_authorization_filters_documentation_and_host_bridge(tmp_path):
    class Example(PythonTool):
        """包括 read 和 secret 两种方法的完整类概览。"""

        def methods(self):
            """只声明可经受控桥接访问的方法。"""
            return {"read": self.read, "secret": self.secret}

        def read(self) -> int:
            """读取固定测试值。

            Returns:
                固定测试值。
            """
            return 7

        def secret(self) -> int:
            """未授权方法的私有说明 PRIVATE_METHOD_DOC。

            Returns:
                不应取得的测试值。
            """
            return 99

    registry = ToolRegistry()
    registry.register_python("Example", Example, methods=("read", "secret"))
    registry.register_stateful(PythonTools)

    async def scenario():
        selected = ["finish", "python__describe_class", "Example.read"]
        async with make_session(tmp_path, registry, selected) as session:
            runtime = session.resource(PythonTools)
            assert "secret" not in session.loop.tools.instructions()
            details = await session.call("python__describe_class", {"class_name": "Example"})
            assert "def read" in details and "secret" not in details
            assert "PRIVATE_METHOD_DOC" not in details
            assert (
                await runtime._call_method(
                    {"class_name": "Example", "method_name": "read", "args": [], "kwargs": {}}
                )
                == 7
            )
            with pytest.raises(PermissionError):
                await runtime._call_method(
                    {"class_name": "Example", "method_name": "secret", "args": [], "kwargs": {}}
                )

    asyncio.run(scenario())


def test_pre_start_factory_reference_cannot_bypass_session_selection(tmp_path):
    calls = []

    def secret() -> int:
        """记录不应由旧工厂引用执行的调用。"""
        calls.append("executed")
        return 1

    registry = ToolRegistry()
    registry.register_callable("secret", secret, description="受限工具。")

    async def scenario():
        session = make_session(tmp_path, registry, ["finish"])
        before_start = session.loop.tools
        async with session:
            with pytest.raises((PermissionError, RuntimeError)):
                await before_start.call("secret", {})
        assert not calls

    asyncio.run(scenario())


def test_closed_session_revokes_previously_borrowed_tool_factory(tmp_path):
    calls = []

    def effect() -> int:
        """记录会话关闭后不应继续执行的普通调用。"""
        calls.append("executed")
        return 1

    registry = ToolRegistry()
    registry.register_callable("effect", effect, description="测试副作用。")

    async def scenario():
        async with make_session(tmp_path, registry) as session:
            borrowed = session.loop.tools
        with pytest.raises(RuntimeError):
            await borrowed.call("effect", {})
        assert not calls

    asyncio.run(scenario())


def test_factory_revocation_cannot_be_undone_by_copy_or_registration():
    factory = ToolFactory()
    factory.register_callable("read", lambda: 3, description="读取测试值。")
    active_copy = factory.copy()
    factory.revoke()
    for operation in (
        factory.copy,
        factory.schemas,
        lambda: factory.register_callable("late", lambda: 4, description="不得重新开放。"),
    ):
        with pytest.raises(RuntimeError):
            operation()
    with pytest.raises(RuntimeError):
        asyncio.run(factory.call("read", {}))
    assert asyncio.run(active_copy.call("read", {})) == 3
