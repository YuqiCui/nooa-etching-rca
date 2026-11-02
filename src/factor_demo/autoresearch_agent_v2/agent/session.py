"""管理会话状态、资源工厂和共享资源的工具实例。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from types import TracebackType
from typing import Any, Self, TypeVar, cast

from ..config import HistoryConfig
from ..telemetry import print_session_startup
from ..tools import StatefulTool, ToolFactory
from .history import compact_history
from .loop import AgentLoop
from .model import Model
from .prompts import SYSTEM_PROMPT
from .typing import DatasetRegistration, SessionState

ResourceT = TypeVar("ResourceT")
StatefulToolT = TypeVar("StatefulToolT", bound=StatefulTool)


class Session:
    """管理连续对话，并作为会话资源与工具实例的统一入口。"""

    def __init__(
        self,
        model: Model,
        state: SessionState,
        *,
        system_prompt: str = SYSTEM_PROMPT,
        history_config: HistoryConfig | None = None,
    ) -> None:
        """创建会话，资源在 start 时才实例化。

        Args:
            model: 现有 AgentLoop 使用的模型。
            state: 对话标识、工作目录和可序列化历史，不存放活资源。
            system_prompt: 各轮模型请求使用的系统提示词。
            history_config: 上下文预算、自动压缩百分比及默认保留轮数。
        """
        self.state = state
        self.loop = AgentLoop(
            model,
            state,
            system_prompt=system_prompt,
            history_config=history_config,
        )
        self._providers: dict[type[Any], Callable[[Session], AbstractAsyncContextManager[Any]]] = {}
        self._instances: dict[type[Any], Any] = {}
        self._stateful_tools: dict[type[StatefulTool], str] = {}
        self._stack = AsyncExitStack()
        self._started = False
        self._starting = False
        self._active = False
        self._closing = False
        # 与当前循环一致，顺序执行工具或对话轮次，关闭等待已开始的操作结束。
        self._operation_lock = asyncio.Lock()
        self.loop.tools.register_callable(
            "register_dataset",
            self.register_dataset,
            description=(
                "登记数据集的名称、文件/目录路径或 URI、简短描述，加入后续请求的提示词。"
                "同名登记更新原条目。只记录信息，不读取、复制或验证数据，不改变访问权限。"
                "相对路径以会话工作目录为基准。"
            ),
            recover_validation_errors=True,
        )

    def register_dataset(self, name: str, path: str, description: str) -> DatasetRegistration:
        """登记数据位置说明，下一次模型请求使用更新后的条目。

        Args:
            name: 当前会话内的数据集名称，同名登记更新原条目。
            path: 文件、目录或 URI；相对路径以 state.workdir 为基准。
            description: 不超过 1000 字符的简短数据说明。

        Returns:
            已保存的名称、位置和描述；不包含读取或验证结果。

        Raises:
            RuntimeError: 会话正在关闭、已经关闭或启动失败。
            ValueError: 字段为空、超长或类型不符合要求。
        """
        if self._closing or (self._started and not (self._starting or self._active)):
            raise RuntimeError("Session is not active")
        dataset = DatasetRegistration(name=name, path=path, description=description)
        self.state.datasets = {**self.state.datasets, dataset.name: dataset}
        return dataset

    def register_resource(
        self,
        resource_type: type[ResourceT],
        factory: Callable[[Session], AbstractAsyncContextManager[ResourceT]],
    ) -> None:
        """登记资源类型及其工厂，不立即创建实例。

        Args:
            resource_type: 资源的类型键，同一会话每种类型只创建一个实例。
            factory: 接收当前会话的异步上下文工厂；按登记顺序执行。

        Raises:
            RuntimeError: 会话已经开始启动或关闭。
            ValueError: 资源类型重复登记。
        """
        if self._started or self._closing:
            raise RuntimeError("Session registrations are frozen")
        if resource_type in self._providers or resource_type in self._stateful_tools:
            raise ValueError(f"Resource already registered: {resource_type.__name__}")
        self._providers[resource_type] = factory

    def register_stateful_tool(self, tool_type: type[StatefulToolT]) -> None:
        """登记有状态工具类，启动时统一调用其 bind 方法。

        Args:
            tool_type: 声明 namespace、bind 和 register_tools 的工具类。

        Raises:
            RuntimeError: 会话已经开始启动或关闭。
            ValueError: 工具类型或名称前缀重复，或名称前缀不合法。
        """
        if self._started or self._closing:
            raise RuntimeError("Session registrations are frozen")
        namespace = tool_type.namespace
        if not namespace.isidentifier() or not namespace.isascii():
            raise ValueError(f"Invalid stateful tool namespace: {namespace!r}")
        if tool_type in self._stateful_tools or tool_type in self._providers:
            raise ValueError(f"Stateful tool already registered: {tool_type.__name__}")
        if namespace in self._stateful_tools.values():
            raise ValueError(f"Stateful tool namespace already registered: {namespace}")
        self._stateful_tools[tool_type] = namespace

    async def bind_resource(
        self,
        resource_type: type[ResourceT],
        factory: Callable[[Session], AbstractAsyncContextManager[ResourceT]],
    ) -> ResourceT:
        """在工具绑定期间取得共享资源，缺失时创建并交给会话清理。

        Args:
            resource_type: 资源类型键，每个会话只保留一个实例。
            factory: 资源缺失时使用的默认异步上下文工厂。
                通过 register_resource 提前登记的宿主工厂优先执行。

        Returns:
            已存在或本次新建的共享资源实例。

        Raises:
            RuntimeError: 当前不在会话启动期间。
            ValueError: 资源类型已被登记为有状态工具。
            TypeError: 工厂返回的实例类型不匹配。
            Exception: 工厂创建失败，原始异常向上传播。
        """
        if not self._starting or self._closing:
            raise RuntimeError("Resources can only be bound during session startup")
        if resource_type in self._stateful_tools:
            raise ValueError("Stateful tools must be bound by Session")
        if resource_type in self._instances:
            return cast(ResourceT, self._instances[resource_type])
        provider = self._providers.get(resource_type, factory)
        instance = await self._stack.enter_async_context(provider(self))
        if not isinstance(instance, resource_type):
            raise TypeError(f"Factory returned wrong type: {resource_type.__name__}")
        self._instances[resource_type] = instance
        return instance

    def resource(self, resource_type: type[ResourceT]) -> ResourceT:
        """取得当前会话已经创建的资源或工具实例。

        Args:
            resource_type: 资源类型或 register_stateful_tool 登记的工具类型。

        Returns:
            当前会话持有的同一实例，不重新调用工厂。

        Raises:
            RuntimeError: 会话不在启动或运行状态，或该依赖尚未创建。
        """
        if self._closing or not (self._starting or self._active):
            raise RuntimeError("Session resources are not active")
        if resource_type not in self._instances:
            raise RuntimeError(f"Resource not ready: {resource_type.__name__}")
        return cast(ResourceT, self._instances[resource_type])

    async def start(self) -> Self:
        """按登记顺序创建资源和工具，全部成功后开放会话。

        Returns:
            当前会话；也可通过 async with 自动启动。

        Raises:
            RuntimeError: 会话启动曾失败或已关闭，不能再次启动。
            Exception: 创建或注册失败，清理已创建资源后向上传播。
        """
        async with self._operation_lock:
            return await self._start_resources()

    async def _start_resources(self) -> Self:
        """在已取得操作锁时启动资源；重复启动运行中的会话不重复构造。"""
        if self._closing:
            raise RuntimeError("Session is closed")
        if self._active:
            return self
        if self._started:
            raise RuntimeError("Session cannot be restarted")
        self._started = True
        self._starting = True
        tools = self.loop.tools.copy()
        try:
            for resource_type, factory in self._providers.items():
                await self.bind_resource(resource_type, factory)
            for tool_type in self._stateful_tools:
                instance = await self._stack.enter_async_context(tool_type.bind(self))
                if not isinstance(instance, tool_type):
                    raise TypeError(f"bind returned wrong type: {tool_type.__name__}")
                self._instances[tool_type] = instance
                instance.register_tools(tools)
        except BaseException:
            try:
                await self._stack.aclose()
            finally:
                self._instances.clear()
            raise
        finally:
            self._starting = False
        self.loop.tools = tools
        self._active = True
        print_session_startup(
            self, (self._instances[tool_type] for tool_type in self._stateful_tools)
        )
        return self

    async def aclose(self) -> None:
        """停止接收操作，等待当前操作退出，并逆序释放会话资源。"""
        self._closing = True
        async with self._operation_lock:
            self._active = False
            # 清除绑定方法引用；清理期间内部实例仍保留至各工厂 finally 完成。
            self.loop.tools = ToolFactory()
            try:
                await self._stack.aclose()
            finally:
                self._instances.clear()

    async def __aenter__(self) -> Self:
        """启动会话并返回当前实例。"""
        return await self.start()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """关闭会话，保留调用方的异常传播。

        Args:
            exc_type: 退出时的异常类型；正常退出时为 None。
            exc: 退出时的异常实例；正常退出时为 None。
            traceback: 异常调用栈；正常退出时为 None。
        """
        await self.aclose()

    def _require_active(self) -> None:
        """拒绝启动前、启动失败后和关闭开始后的新操作。"""
        if not self._active or self._closing:
            raise RuntimeError("Session is not active")

    def schemas(self) -> list[dict[str, Any]]:
        """返回当前会话的模型工具描述。

        Returns:
            现有 ToolFactory 格式的工具描述列表。

        Raises:
            RuntimeError: 会话未处于运行状态。
        """
        self._require_active()
        return self.loop.tools.schemas()

    async def call(
        self, name: str, arguments: dict[str, Any], *, call_id: str | None = None
    ) -> Any:
        """直接调用会话中的工具，关闭时等待本次调用退出。

        Args:
            name: 已注册的工具名称。
            arguments: 工具关键字参数。
            call_id: 可选的宿主执行标识，用于同一次执行的重试去重。

        Returns:
            工具的原始返回值。

        Raises:
            RuntimeError: 会话未处于运行状态。
            Exception: 工具错误向上传播。
        """
        async with self._operation_lock:
            self._require_active()
            return await self.loop.tools.call(name, arguments, call_id=call_id)

    async def compact(self, *, keep_recent_turns: int | None = None) -> bool:
        """压缩较早的对话，与运行及工具操作共用会话锁。

        Args:
            keep_recent_turns: 本次覆盖配置中的保留轮数；None 使用配置，0 总结全部历史。

        Returns:
            保存新摘要时返回 True；没有可新增压缩的历史时返回 False。

        Raises:
            RuntimeError: 会话已经关闭、启动失败或历史被外部并发修改。
            ValueError: 参数、历史协议或模型返回的摘要不合法。
            Exception: 启动、模型请求或取消错误向上传播，原压缩检查点保持不变。
        """
        async with self._operation_lock:
            await self._start_resources()
            self._require_active()
            return await compact_history(
                self.state,
                self.loop.model,
                system_prompt=self.loop.build_system_prompt(),
                history_config=self.loop.history_config,
                keep_recent_turns=keep_recent_turns,
            )

    async def run(self, content: str) -> str:
        """通过现有 AgentLoop 执行一轮对话，轮次结束后继续保留资源。

        Args:
            content: 本轮用户输入。

        Returns:
            本轮 finish 工具的最终输出。

        Raises:
            RuntimeError: 会话已关闭或先前启动失败。
            Exception: 资源启动或循环错误向上传播。
        """
        async with self._operation_lock:
            await self._start_resources()
            self._require_active()
            return await self.loop.run(content)
