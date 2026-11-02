"""验证会话资源工厂、工具组共享和生命周期。"""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from pydantic import BaseModel, ValidationError

from factor_demo.autoresearch_agent_v2.agent import Session, SessionState
from factor_demo.autoresearch_agent_v2.agent.model import Model
from factor_demo.autoresearch_agent_v2.config import ModelConfig
from factor_demo.autoresearch_agent_v2.sandboxing import Sandbox, SandboxConfig
from factor_demo.autoresearch_agent_v2.tools import SandboxTools, ToolFactory


class SharedResource:
    """测试用的会话资源，包含共享选择和可观察的临时目录。"""

    def __init__(self, root):
        """保存目录和当前选择。"""
        self.root = root
        self.value = 0

    @staticmethod
    @asynccontextmanager
    async def create(session):
        """创建可实际回收的临时目录。"""
        with TemporaryDirectory(dir=session.state.workdir) as directory:
            yield SharedResource(Path(directory))


class SharedTools:
    """测试用的工具组，两个方法引用同一会话资源。"""

    namespace = "shared"

    def __init__(self, resource):
        """绑定共享资源。"""
        self.resource = resource

    @classmethod
    @asynccontextmanager
    async def bind(cls, session):
        """从会话取得或创建共享依赖资源。"""
        resource = await session.bind_resource(SharedResource, SharedResource.create)
        yield cls(resource)

    def register_tools(self, factory):
        """绑定两个实例方法。"""
        factory.register_callable(
            f"{self.namespace}__set", self.set_value, description="修改共享值。"
        )
        factory.register_callable(
            f"{self.namespace}__get", self.get_value, description="查看共享值。"
        )

    def set_value(self, value: int) -> int:
        """修改共享值。"""
        self.resource.value = value
        return value

    def get_value(self) -> int:
        """读取共享值。"""
        return self.resource.value


def make_session(tmp_path, identity="first", *, resources=True):
    """创建使用占位模型的会话，不发送模型请求。"""
    model = Model(ModelConfig(model="openai/fixture-model", api_key="fixture-key"))
    state = SessionState(session_id=identity, agent_id=identity, task_id="task", workdir=tmp_path)
    session = Session(model, state)
    if resources:
        session.register_stateful_tool(SharedTools)
    return session


def test_lazy_registration_shared_instances_and_session_isolation(tmp_path):
    """启动才创建资源，同会话共享，两个会话保持独立。"""

    async def scenario():
        first = make_session(tmp_path, "A")
        second = make_session(tmp_path, "B")
        assert list(tmp_path.iterdir()) == []
        with pytest.raises(RuntimeError, match="not active"):
            first.resource(SharedResource)
        async with first, second:
            resource = first.resource(SharedResource)
            assert first.resource(SharedTools).resource is resource
            assert second.resource(SharedResource) is not resource
            await first.start()
            assert first.resource(SharedResource) is resource
            assert len(list(tmp_path.iterdir())) == 2
            await first.call("shared__set", {"value": 7})
            assert await first.call("shared__get", {}) == 7
            assert await second.call("shared__get", {}) == 0
        assert list(tmp_path.iterdir()) == []

    asyncio.run(scenario())


def test_multiple_tool_classes_share_one_resource(tmp_path):
    """不同工具类通过同一类型键取得一份资源。"""

    class OtherTools(SharedTools):
        """使用独立类型键的第二组工具。"""

        namespace = "other"

    async def scenario():
        session = make_session(tmp_path)
        session.register_stateful_tool(OtherTools)
        async with session:
            assert session.resource(SharedTools).resource is session.resource(OtherTools).resource
            assert len(list(tmp_path.iterdir())) == 1
            await session.call("shared__set", {"value": 4})
            assert await session.call("other__get", {}) == 4

    asyncio.run(scenario())


def test_session_calls_each_bind_once_and_releases_tools_before_shared_resource(tmp_path):
    """按工具登记顺序绑定一次，关闭时先释放工具再释放其共享资源。"""
    events = []

    @asynccontextmanager
    async def resource_factory(session):
        async with SharedResource.create(session) as resource:
            events.append("resource created")
            try:
                yield resource
            finally:
                events.append("resource closed")

    class FirstTools(SharedTools):
        """记录绑定和清理顺序。"""

        namespace = "first"

        @classmethod
        @asynccontextmanager
        async def bind(cls, session):
            """请求同一份资源，并记录会话释放工具的顺序。"""
            events.append(f"{cls.namespace} bind")
            resource = await session.bind_resource(SharedResource, resource_factory)
            try:
                yield cls(resource)
            finally:
                events.append(f"{cls.namespace} closed")

    class SecondTools(FirstTools):
        """通过独立类型键复用同一底层资源。"""

        namespace = "second"

    async def scenario():
        session = make_session(tmp_path, resources=False)
        session.register_stateful_tool(FirstTools)
        session.register_stateful_tool(SecondTools)
        with pytest.raises(RuntimeError, match="startup"):
            await session.bind_resource(SharedResource, resource_factory)
        async with session:
            await session.start()
            assert session.resource(FirstTools).resource is session.resource(SecondTools).resource
            assert len(list(tmp_path.iterdir())) == 1
            assert events == ["first bind", "resource created", "second bind"]
        assert list(tmp_path.iterdir()) == []

    asyncio.run(scenario())
    assert events == [
        "first bind",
        "resource created",
        "second bind",
        "second closed",
        "first closed",
        "resource closed",
    ]


@pytest.mark.parametrize("failure", ["error", "cancel"])
def test_failed_bind_cleans_resources_created_before_the_tool_yields(tmp_path, failure):
    """bind 在返回实例前失败或被取消，已经纳管的共享资源仍会回收。"""

    class FailingTools(SharedTools):
        """获取资源后故意终止绑定。"""

        @classmethod
        @asynccontextmanager
        async def bind(cls, session):
            """在资源创建之后触发异常或任务取消。"""
            resource = await session.bind_resource(SharedResource, SharedResource.create)
            if failure == "error":
                raise ValueError("binding failed")
            task = asyncio.current_task()
            assert task is not None
            task.cancel()
            await asyncio.sleep(0)
            yield cls(resource)

    async def scenario():
        session = make_session(tmp_path, resources=False)
        session.register_stateful_tool(FailingTools)
        expected = ValueError if failure == "error" else asyncio.CancelledError
        with pytest.raises(expected):
            await asyncio.create_task(session.start())
        assert list(tmp_path.iterdir()) == []
        with pytest.raises(RuntimeError, match="not active"):
            session.resource(SharedResource)

    asyncio.run(scenario())


def test_start_preserves_tools_registered_before_resource_initialization(tmp_path):
    """保留已有 session.loop.tools.register 的使用方式。"""

    async def scenario():
        session = make_session(tmp_path)
        session.loop.tools.register(
            "custom", lambda: "kept", description="已有工具。", parameters={}
        )
        async with session:
            assert await session.call("custom", {}) == "kept"
            assert await session.call("shared__get", {}) == 0

    asyncio.run(scenario())


def test_run_starts_resources_and_preserves_them_across_real_loop_turns(monkeypatch, tmp_path):
    """模拟模型响应驱动实际循环，首次 run 自动启动，后续轮次复用实例和历史。"""
    requests = []

    async def completion(self, messages, *, tools=None):
        requests.append((messages, tools))
        calls = []
        if len(requests) == 1:
            calls.append(
                {
                    "id": "set",
                    "type": "function",
                    "function": {"name": "shared__set", "arguments": '{"value":7}'},
                }
            )
        calls.append(
            {
                "id": f"finish-{len(requests)}",
                "type": "function",
                "function": {"name": "finish", "arguments": '{"content":"done"}'},
            }
        )
        return {"role": "assistant", "tool_calls": calls}

    monkeypatch.setattr(Model, "completion", completion)

    async def scenario():
        session = make_session(tmp_path)
        try:
            assert await session.run("第一轮") == "done"
            resource = session.resource(SharedResource)
            assert await session.run("第二轮") == "done"
            assert session.resource(SharedResource) is resource
            assert resource.value == 7
            assert [
                message["content"]
                for message in session.state.messages
                if message["role"] == "user"
            ] == ["第一轮", "第二轮"]
            assert "shared__set" in [item["function"]["name"] for item in requests[0][1]]
        finally:
            await session.aclose()
        assert not resource.root.exists()

    asyncio.run(scenario())


def test_duplicate_registration_freeze_and_argument_validation(tmp_path):
    """重复类型与组名报错，启动后禁止登记，参数校验失败不改变共享值。"""

    class OtherTools(SharedTools):
        """用于检查组名冲突的独立类型。"""

    class BadNamespaceTools(SharedTools):
        """提供无效的名称前缀。"""

        namespace = "bad-name"

    async def scenario():
        session = make_session(tmp_path)
        session.register_resource(SharedResource, SharedResource.create)
        with pytest.raises(ValueError, match="Resource already registered"):
            session.register_resource(SharedResource, SharedResource.create)
        with pytest.raises(ValueError, match="Stateful tool already registered"):
            session.register_stateful_tool(SharedTools)
        with pytest.raises(ValueError, match="namespace already registered"):
            session.register_stateful_tool(OtherTools)
        with pytest.raises(ValueError, match="Invalid stateful tool namespace"):
            session.register_stateful_tool(BadNamespaceTools)
        async with session:
            with pytest.raises(RuntimeError, match="frozen"):
                session.register_stateful_tool(OtherTools)
            for arguments in [{"value": "7"}, {"value": 7, "extra": True}]:
                with pytest.raises(ValidationError):
                    await session.call("shared__set", arguments)
            assert await session.call("shared__get", {}) == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("exit_kind", ["normal", "error", "cancel"])
def test_scope_exit_releases_resources_and_rejects_later_access(tmp_path, exit_kind):
    """正常退出、会话异常和任务取消均回收资源。"""

    async def scenario():
        session = make_session(tmp_path)
        roots = []

        async def run_scope():
            async with session:
                roots.append(session.resource(SharedResource).root)
                if exit_kind == "error":
                    raise ValueError("session failed")
                if exit_kind == "cancel":
                    task = asyncio.current_task()
                    assert task is not None
                    task.cancel()
                    await asyncio.sleep(0)

        if exit_kind == "normal":
            await run_scope()
        else:
            with pytest.raises(ValueError if exit_kind == "error" else asyncio.CancelledError):
                await asyncio.create_task(run_scope())
        assert not roots[0].exists()
        await session.aclose()
        with pytest.raises(RuntimeError, match="not active"):
            session.resource(SharedResource)
        with pytest.raises(RuntimeError, match="not active"):
            await session.call("shared__get", {})
        with pytest.raises(RuntimeError, match="closed"):
            await session.run("after close")

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["open", "register", "type"])
def test_failed_startup_rolls_back_in_reverse_order(tmp_path, failure):
    """工厂创建、工具绑定或返回类型错误时，逆序清理且不开放部分资源。"""
    events = []

    class BrokenTools:
        """用于触发工具名冲突。"""

        namespace = "broken"

        @classmethod
        @asynccontextmanager
        async def bind(cls, session):
            """通过可控的错误上下文绑定当前工具。"""
            async with broken_factory(session) as instance:
                yield instance

        def register_tools(self, factory):
            """尝试覆盖内置 finish。"""
            factory.register(lambda: None, description="重复工具", parameters={})

    @asynccontextmanager
    async def resource_factory(session):
        async with SharedResource.create(session) as resource:
            events.append("resource opened")
            try:
                yield resource
            finally:
                events.append("resource closed")

    @asynccontextmanager
    async def broken_factory(session):
        assert session.resource(SharedResource).root.is_dir()
        events.append("tools opened")
        try:
            if failure == "open":
                raise ValueError("resource unavailable")
            yield object() if failure == "type" else BrokenTools()
        finally:
            events.append("tools closed")

    async def scenario():
        session = make_session(tmp_path, resources=False)
        session.register_resource(SharedResource, resource_factory)
        session.register_stateful_tool(BrokenTools)
        with pytest.raises((ValueError, TypeError)):
            await session.start()
        assert list(tmp_path.iterdir()) == []
        with pytest.raises(RuntimeError, match="not active"):
            session.resource(SharedResource)
        with pytest.raises(RuntimeError, match="cannot be restarted"):
            await session.start()

    asyncio.run(scenario())
    assert events == ["resource opened", "tools opened", "tools closed", "resource closed"]


def test_host_resource_factories_run_before_tool_binding_regardless_of_registration_order(tmp_path):
    """先执行全部宿主资源工厂，工具绑定时复用宿主提供的实例。"""

    async def scenario():
        session = make_session(tmp_path, resources=False)
        session.register_stateful_tool(SharedTools)
        session.register_resource(SharedResource, SharedResource.create)
        async with session:
            assert session.resource(SharedTools).resource is session.resource(SharedResource)
            assert len(list(tmp_path.iterdir())) == 1

    asyncio.run(scenario())


def test_host_factory_overrides_default_even_when_an_earlier_factory_requests_it(tmp_path):
    """宿主工厂优先于默认工厂，不受其他资源工厂的依赖请求顺序影响。"""

    class Consumer:
        """引用共享资源的宿主对象。"""

        def __init__(self, resource):
            """保存宿主选择的资源。"""
            self.resource = resource

    @asynccontextmanager
    async def create_consumer(session):
        resource = await session.bind_resource(SharedResource, SharedResource.create)
        yield Consumer(resource)

    @asynccontextmanager
    async def host_factory(session):
        async with SharedResource.create(session) as resource:
            resource.value = 77
            yield resource

    async def scenario():
        session = make_session(tmp_path, resources=False)
        session.register_resource(Consumer, create_consumer)
        session.register_resource(SharedResource, host_factory)
        session.register_stateful_tool(SharedTools)
        async with session:
            resource = session.resource(SharedResource)
            assert session.resource(Consumer).resource is resource
            assert session.resource(SharedTools).resource is resource
            assert resource.value == 77
            assert len(list(tmp_path.iterdir())) == 1

    asyncio.run(scenario())


def test_close_waits_for_running_operation_and_rejects_queued_call(tmp_path):
    """关闭等待当前调用完成，并拒绝排队中的新调用。"""

    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()

        async def slow():
            started.set()
            await release.wait()
            return "done"

        session = make_session(tmp_path)
        await session.start()
        root = session.resource(SharedResource).root
        session.loop.tools.register_callable("slow", slow, description="等待放行。")
        running = asyncio.create_task(session.call("slow", {}))
        await started.wait()
        queued = asyncio.create_task(session.call("slow", {}))
        closing = asyncio.create_task(session.aclose())
        await asyncio.sleep(0)
        assert not closing.done()
        assert root.is_dir()
        release.set()
        assert await running == "done"
        with pytest.raises(RuntimeError, match="not active"):
            await queued
        await closing
        assert not root.exists()

    asyncio.run(asyncio.wait_for(scenario(), timeout=5))


@pytest.mark.parametrize("configured", [False, True])
def test_sandbox_tools_use_the_same_bind_protocol_and_keep_host_config(tmp_path, configured):
    """沙箱工具通过统一 bind 协议获得资源，保留宿主配置并在会话结束时关闭。"""

    async def scenario():
        session = make_session(tmp_path, resources=False)
        if configured:
            SandboxTools.register(session, config=SandboxConfig(workdir=tmp_path, timeout_s=8))
        else:
            session.register_stateful_tool(SandboxTools)
        async with session:
            sandbox = session.resource(Sandbox)
            assert session.resource(SandboxTools).sandbox is sandbox
            assert sandbox.config.timeout_s == (8 if configured else 120)
            names = [item["function"]["name"] for item in session.schemas()]
            assert names == ["finish", "register_dataset", "sandbox__run_shell"]
        # 已关闭检查在创建进程之前执行，本用例无需启动本机执行后端。
        with pytest.raises(RuntimeError, match="closed"):
            await sandbox.run_shell("true")

    asyncio.run(scenario())


def test_sync_and_async_model_results_are_json_serializable():
    """同步和异步工具返回 Pydantic 模型时，统一转换成可写入历史的字典。"""

    class Result(BaseModel):
        """带结构的工具结果。"""

        value: int

    def sync_tool(value: int) -> Result:
        """返回同步模型结果。"""
        return Result(value=value)

    async def async_tool(value: int) -> Result:
        """返回异步模型结果。"""
        return Result(value=value)

    async def scenario():
        factory = ToolFactory()
        factory.register_callable("sync", sync_tool, description="同步工具。")
        factory.register_callable("async", async_tool, description="异步工具。")
        assert await factory.call("sync", {"value": 1}) == {"value": 1}
        assert await factory.call("async", {"value": 2}) == {"value": 2}

    asyncio.run(scenario())
