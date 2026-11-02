"""声明 Python 工具类、显式方法清单及统一注册入口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, ClassVar, Self, TypeVar

if TYPE_CHECKING:
    from ..agent.session import Session

ToolT = TypeVar("ToolT", bound="PythonTool")


class PythonTool(ABC):
    """按类提供文档和方法，实例状态仅属于当前会话。"""

    _registry: ClassVar[dict[str, type[PythonTool]]] = {}
    """显式登记的工具类，不在登记阶段创建实例或读取方法 Schema。"""

    @classmethod
    def register(cls, name: str) -> Callable[[type[ToolT]], type[ToolT]]:
        """用装饰器登记可在 Python 中使用的工具类。

        Args:
            name: 提供给模型的类名，必须是非保留的 ASCII Python 标识符。

        Returns:
            校验后原样返回实现类的装饰器。

        Raises:
            ValueError: 名称无效、重复或类没有独立编写的文档字符串。
            TypeError: 实现不是当前工具基类的具体子类。
        """
        from inspect import isabstract
        from keyword import iskeyword

        def register_tool(candidate: type[ToolT]) -> type[ToolT]:
            """检查名称和实现后保存类，不执行构造函数。"""
            if (
                not name.isidentifier()
                or not name.isascii()
                or name.startswith("_")
                or iskeyword(name)
                or name in {"print", "input", "__builtins__"}
            ):
                raise ValueError(f"无效的 Python 工具类名：{name!r}")
            if name in cls._registry:
                raise ValueError(f"Python 工具类已经注册：{name}")
            if (
                not isinstance(candidate, type)
                or not issubclass(candidate, cls)
                or isabstract(candidate)
            ):
                raise TypeError("Python 工具必须是 PythonTool 的具体子类。")
            if not candidate.__doc__ or not candidate.__doc__.strip():
                raise ValueError("Python 工具类必须编写功能和方法概览。")
            cls._registry[name] = candidate
            return candidate

        return register_tool

    @classmethod
    def registered(cls) -> dict[str, type[PythonTool]]:
        """获取当前显式登记的类清单。

        Returns:
            独立字典；调用方不能通过它修改全局注册表。
        """
        return cls._registry.copy()

    @classmethod
    @asynccontextmanager
    async def bind(cls, session: Session) -> AsyncIterator[Self]:
        """查看详情时创建会话实例，子类可重写以绑定已准备好的资源。

        Args:
            session: 当前运行中的会话；可通过 resource 取得已启动的共享资源。

        Yields:
            默认无参构造的工具实例；有资源的子类应在 finally 中负责清理。
        """
        yield cls()

    @abstractmethod
    def methods(self) -> dict[str, Callable[..., Any]]:
        """明确列出允许 agent 调用的方法，不自动暴露其他公有成员。

        Returns:
            Python 方法名到当前实例绑定方法的映射。
        """
        raise NotImplementedError
