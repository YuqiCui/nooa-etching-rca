"""按名称注册工具，并提供模型调用所需的描述。"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractAsyncContextManager, contextmanager
from copy import deepcopy
from functools import wraps
from inspect import isawaitable
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, Self

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from ..errors import UnknownToolError
from .builtin_tools.messages import finish
from .call_context import execution_id
from .policy import ToolPolicy

if TYPE_CHECKING:
    from ..agent.session import Session


class StatefulTool(Protocol):
    """约定有状态工具的会话绑定、共享资源和方法注册。"""

    namespace: ClassVar[str]
    """当前工具类的名称前缀，在同一会话中必须唯一。"""

    @classmethod
    def bind(cls, session: Session) -> AbstractAsyncContextManager[Self]:
        """创建会话专属实例，并通过上下文释放其拥有的资源。

        Args:
            session: 正在启动的会话，提供共享资源入口。

        Returns:
            进入时返回工具实例、退出时执行清理的异步上下文。
        """
        ...

    def register_tools(self, factory: ToolFactory) -> None:
        """绑定当前实例提供的工具。

        Args:
            factory: 当前会话的工具注册表。
        """
        ...


class ToolFactory:
    """保存当前 Agent 可调用的函数及其 JSON Schema。"""

    def __init__(self) -> None:
        """创建空的工具注册表。"""
        self._functions: dict[str, Callable[..., Any]] = {}
        self._schemas: dict[str, dict[str, Any]] = {}
        self._instructions: list[tuple[tuple[str, ...], str]] = []
        self.policy: ToolPolicy | None = None
        self._registration_names: tuple[str, ...] = ()
        self._revoked = False

    def add_instructions(self, content: str, *, names: Sequence[str] | None = None) -> None:
        """追加与当前工具表一起提供给模型的使用说明。

        Args:
            content: 不含凭据或宿主内部状态的工具使用说明。
            names: 说明所依赖的工具名；缺省时使用当前登记组的静态清单。
        """
        self._require_open()
        required = tuple(names) if names is not None else self._registration_names
        self._instructions.append((required, content))

    def instructions(self) -> str:
        """返回拼接后的工具说明。

        Returns:
            默认提示词之后追加的文本；没有说明时为空字符串。
        """
        self._require_open()
        return "\n\n".join(
            content
            for names, content in self._instructions
            if self.policy is None or all(self.policy.allows(name) for name in names)
        )

    def names(self) -> tuple[str, ...]:
        """返回已登记名称，用于会话启动前解析工具策略。"""
        self._require_open()
        return tuple(self._functions)

    def restrict(self, policy: ToolPolicy) -> None:
        """将同一授权结果应用到 Schema、说明和真实调用。"""
        self._require_open()
        self.policy = policy

    @contextmanager
    def registration_scope(self, names: Sequence[str]) -> Iterator[None]:
        """将工具组说明与其公开名称绑定，避免暴露未授权接口。"""
        self._require_open()
        previous = self._registration_names
        self._registration_names = tuple(names)
        try:
            yield
        finally:
            self._registration_names = previous

    def register(
        self,
        name: str,
        function: Callable[..., Any],
        *,
        description: str,
        parameters: dict[str, Any],
    ) -> None:
        """注册一个同步或异步工具。

        Args:
            name: 模型调用工具时使用的名称。
            function: 接收工具关键字参数的函数。
            description: 提供给模型的工具用途说明。
            parameters: 工具参数的 JSON Schema。

        Raises:
            ValueError: 名称已被注册。
        """
        self._require_open()
        if name in self._functions:
            raise ValueError(f"Tool already registered: {name}")
        if self.policy is not None:
            self.policy.include_registered(name)
        self._functions[name] = function
        self._schemas[name] = {
            "type": "function",
            "function": {"name": name, "description": description, "parameters": parameters},
        }

    def register_callable(
        self,
        name: str,
        function: Callable[..., Any],
        *,
        description: str,
        recover_validation_errors: bool = False,
    ) -> None:
        """由函数签名生成参数 Schema，并在调用前校验参数。

        Args:
            name: 模型调用工具时使用的名称。
            function: 参数和返回值具有类型注解的同步或异步函数。
            description: 提供给模型的用途说明。
            recover_validation_errors: 为 True 时将参数错误作为简短工具结果返回，
                不回显输入；默认保持抛出 ValidationError 的行为。

        Raises:
            ValueError: 工具名称重复。
            Exception: 函数注解无法生成参数校验器或 Schema。
        """

        # 保留原模块、签名和注解，确保跨模块的绑定方法也能解析延迟注解。
        @wraps(function)
        def bound_function(**arguments: Any) -> Any:
            """将实例方法转为保留原签名的普通可调用函数。"""
            return function(**arguments)

        adapter = TypeAdapter(bound_function, config=ConfigDict(extra="forbid", strict=True))

        def invoke(**arguments: Any) -> Any:
            """校验参数后调用原始函数，异步结果由统一调用入口等待。"""
            try:
                return adapter.validate_python(arguments)
            except ValidationError:
                if not recover_validation_errors:
                    raise
                return {
                    "error": "invalid_argument",
                    "hint": "Use the declared parameter names, types and bounds.",
                }

        self.register(name, invoke, description=description, parameters=adapter.json_schema())

    def schemas(self) -> list[dict[str, Any]]:
        """返回模型使用的工具描述。

        Returns:
            按注册顺序排列的工具描述列表。
        """
        self._require_open()
        return [
            schema
            for name, schema in self._schemas.items()
            if self.policy is None or self.policy.allows(name)
        ]

    def copy(self) -> ToolFactory:
        """复制注册表，保留已有函数绑定，并独立保存 Schema。

        Returns:
            可用于会话启动期间注册工具组的独立工具表。
        """
        self._require_open()
        factory = ToolFactory()
        factory._functions = self._functions.copy()
        factory._schemas = deepcopy(self._schemas)
        factory._instructions = self._instructions.copy()
        factory.policy = self.policy
        return factory

    async def call(
        self, name: str, arguments: dict[str, Any], *, call_id: str | None = None
    ) -> Any:
        """执行已注册工具，并等待异步结果。

        Args:
            name: 已注册的工具名称。
            arguments: 传入函数的关键字参数。
            call_id: 宿主执行标识，同一次执行重试时保持不变。

        Returns:
            工具函数的返回值。

        Raises:
            UnknownToolError: 工具名称未注册。
            Exception: 工具执行错误原样向上传播。
        """
        self._require_open()
        try:
            function = self._functions[name]
        except KeyError as error:
            raise UnknownToolError(f"Unknown tool: {name}. Use a declared tool name.") from error
        if self.policy is not None:
            self.policy.require(name)
        token = execution_id.set(call_id)
        try:
            result = function(**arguments)
            if isawaitable(result):
                result = await result
            if isinstance(result, BaseModel):
                return result.model_dump(mode="json")
            return result
        finally:
            execution_id.reset(token)

    def revoke(self) -> None:
        """撤销借出的调用入口，释放闭包引用；副本生命周期互不影响。"""
        self._revoked = True
        self._functions.clear()
        self._schemas.clear()
        self._instructions.clear()

    def _require_open(self) -> None:
        """阻止关闭后的调用、复制或重新登记恢复已撤销能力。"""
        if self._revoked:
            raise RuntimeError("工具工厂已关闭")


def create_tool_factory() -> ToolFactory:
    """创建工厂并注册不依赖会话资源的内置工具。

    Returns:
        已注册 finish 的独立工具工厂。
    """
    factory = ToolFactory()
    parameters = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"content": {"type": "string"}},
        "required": ["content"],
    }
    factory.register(
        "finish",
        finish,
        description="结束当前分配的工作，内容由运行时交给父会话或用户。",
        parameters=parameters,
    )
    return factory
