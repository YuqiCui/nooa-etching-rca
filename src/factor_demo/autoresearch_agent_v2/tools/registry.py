"""保存应用显式登记的工具定义，不创建资源或自动导入业务包。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..agent.session import Session
    from ..python_tools.base import PythonTool
    from .base import StatefulTool

ToolKind = Literal["builtin", "business"]


@dataclass(frozen=True)
class PythonToolRegistration:
    """保存 Python 类与无需实例化即可检查的公开方法清单。"""

    tool_type: type[PythonTool]
    methods: tuple[str, ...]
    kind: ToolKind


@dataclass(frozen=True)
class StatefulRegistration:
    """将工具组的公开名称与延迟绑定类型关联。"""

    tool_type: type[StatefulTool]
    names: tuple[str, ...]
    kind: ToolKind


@dataclass(frozen=True)
class CallableRegistration:
    """保存无需会话资源的函数及其公开说明。"""

    name: str
    function: Callable[..., Any]
    description: str
    kind: ToolKind


class ToolRegistry:
    """每个应用装配持有独立注册目录，会话只取得定义快照。"""

    def __init__(self) -> None:
        """创建没有任何隐式业务能力的注册目录。"""
        self._callables: dict[str, CallableRegistration] = {}
        self._stateful: dict[type[StatefulTool], StatefulRegistration] = {}
        self._python: dict[str, PythonToolRegistration] = {}
        self._names: set[str] = set()

    def _reserve(self, names: Sequence[str]) -> tuple[str, ...]:
        """在修改目录之前检查整组名称，避免部分登记。"""
        if isinstance(names, str):
            raise TypeError("工具清单必须是名称序列，不能是单个字符串")
        declared = tuple(names)
        if not declared or any(not isinstance(name, str) or not name for name in declared):
            raise ValueError("工具必须声明非空的公开名称")
        if len(set(declared)) != len(declared) or self._names.intersection(declared):
            raise ValueError("工具公开名称重复")
        self._names.update(declared)
        return declared

    def register_callable(
        self,
        name: str,
        function: Callable[..., Any],
        *,
        description: str,
        kind: ToolKind = "business",
    ) -> None:
        """登记普通函数，不执行函数或创建资源。"""
        self._validate_kind(kind)
        self._reserve((name,))
        self._callables[name] = CallableRegistration(name, function, description, kind)

    def register_stateful(
        self,
        tool_type: type[StatefulTool],
        *,
        names: Sequence[str] | None = None,
        kind: ToolKind = "builtin",
    ) -> None:
        """登记延迟绑定的工具组及其静态公开清单。"""
        self._validate_kind(kind)
        namespace = getattr(tool_type, "namespace", None)
        if (
            not isinstance(namespace, str)
            or not namespace.isidentifier()
            or not namespace.isascii()
        ):
            raise ValueError("工具组必须声明合法的 namespace")
        if any(entry.tool_type.namespace == namespace for entry in self._stateful.values()):
            raise ValueError(f"工具组 namespace 重复：{namespace}")
        if tool_type in self._stateful:
            raise ValueError(f"工具组已登记：{tool_type.__name__}")
        declared = self._reserve(names if names is not None else self.tool_names(tool_type))
        self._stateful[tool_type] = StatefulRegistration(tool_type, declared, kind)

    def register_python(
        self,
        name: str,
        tool_type: type[PythonTool],
        *,
        methods: Sequence[str] | None = None,
        kind: ToolKind = "business",
    ) -> None:
        """登记 Python 工具类，方法清单不依赖实例构造。"""
        from ..python_tools.base import validate_tool_definition

        self._validate_kind(kind)
        validate_tool_definition(name, tool_type)
        if name in self._python:
            raise ValueError(f"Python 工具类已登记：{name}")
        if isinstance(methods, str):
            raise TypeError("方法清单必须是名称序列，不能是单个字符串")
        declared = tuple(methods or getattr(tool_type, "tool_methods", ()))
        if not declared or any(
            not isinstance(method, str) or not method.isidentifier() or method.startswith("_")
            for method in declared
        ):
            raise ValueError("Python 工具类必须声明公开方法清单")
        self._reserve(tuple(f"{name}.{method}" for method in declared))
        self._python[name] = PythonToolRegistration(tool_type, declared, kind)

    def tool_names(self, tool_type: type[StatefulTool]) -> tuple[str, ...]:
        """取得工具组的静态名称；旧工具组可在 all 模式下兼容绑定。"""
        entry = self._stateful.get(tool_type)
        return entry.names if entry is not None else tuple(getattr(tool_type, "tool_names", ()))

    @staticmethod
    def _validate_kind(kind: str) -> None:
        """工具来源只接受明确声明的两种分类。"""
        if kind not in {"builtin", "business"}:
            raise ValueError("工具来源必须是 builtin 或 business")

    def names(self) -> tuple[str, ...]:
        """返回全部公开名称，包括 Python 类方法。"""
        return tuple(sorted(self._names))

    def providers(self) -> dict[str, tuple[str, ...]]:
        """生成显式 provider 选择名称，不支持隐式目录扫描。"""
        providers = {
            f"{entry.kind}:{entry.tool_type.namespace}": entry.names
            for entry in self._stateful.values()
        }
        providers.update(
            {
                f"{entry.kind}:{name}": tuple(f"{name}.{method}" for method in entry.methods)
                for name, entry in self._python.items()
            }
        )
        return providers

    def python_tools(self) -> dict[str, PythonToolRegistration]:
        """提供定义快照，调用方不能修改注册目录。"""
        return self._python.copy()

    def install(self, session: Session) -> None:
        """在会话启动前登记函数与工具组，实例仍由会话创建。"""
        for entry in self._callables.values():
            session.loop.tools.register_callable(
                entry.name, entry.function, description=entry.description
            )
        for entry in self._stateful.values():
            session.register_stateful_tool(entry.tool_type)

    def copy(self) -> ToolRegistry:
        """复制注册定义，后续登记互不影响。"""
        copied = ToolRegistry()
        copied._callables = self._callables.copy()
        copied._stateful = self._stateful.copy()
        copied._python = self._python.copy()
        copied._names = self._names.copy()
        return copied


def register_builtin_tools(registry: ToolRegistry) -> None:
    """显式启用文件、Notebook/Python 桥接与隔离 Shell 通用能力。"""
    from ..python_tools.runtime import PythonTools
    from .builtin_tools.files import FileTools
    from .builtin_tools.sandbox import SandboxTools

    for tool_type in (FileTools, PythonTools, SandboxTools):
        registry.register_stateful(tool_type, kind="builtin")
