"""将类级渐进发现、按需实例化和 Python 工具执行绑定到 Session。"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from keyword import iskeyword
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from ..sandboxing import Sandbox
from ..tools.base import ToolFactory
from ..tools.builtin_tools.notebooks import NotebookTools
from ..tools.builtin_tools.notebooks.kernel import NotebookKernel
from ..tools.builtin_tools.sandbox import SandboxTools
from .base import PythonTool
from .rendering import render_class_documentation
from .schema import ToolMethod
from .typing import ClassDetails, PythonToolCall

if TYPE_CHECKING:
    from ..agent.session import Session


class PythonTools(NotebookTools):
    """为会话提供类目录、详情加载与 Notebook 中的 Python 调用。"""

    namespace: ClassVar[str] = "python"
    """类发现入口的前缀；代码执行统一使用 notebook 工具。"""

    def __init__(self, session: Session, sandbox: Sandbox) -> None:
        """固定当前会话的目录快照，不实例化业务工具。

        Args:
            session: 持有授权资源的当前会话。
            sandbox: 当前会话共用的固定权限执行环境。
        """
        super().__init__(sandbox)
        self._session = session
        self._classes = PythonTool.registered()
        self._loaded: dict[str, dict[str, ToolMethod]] = {}
        self._details: dict[str, str] = {}
        self._stack = AsyncExitStack()
        self._closed = False

    def class_names(self) -> tuple[str, ...]:
        """返回当前会话的 Python 工具类目录，不加载详情或创建实例。

        Returns:
            按注册顺序排列的类名，使用会话绑定时保存的快照。
        """
        return tuple(self._classes)

    @classmethod
    @asynccontextmanager
    async def bind(cls, session: Session) -> AsyncIterator[PythonTools]:
        """共享会话沙箱，在会话结束时关闭已按需创建的工具实例。

        Args:
            session: 正在绑定资源的会话。

        Yields:
            与其他会话隔离加载状态的 Python 工具入口。

        Raises:
            ValueError: 宿主沙箱的工作目录与会话不匹配。
        """
        sandbox = await session.bind_resource(Sandbox, SandboxTools.create_sandbox)
        SandboxTools._check_workdir(session, sandbox.config)
        instance = cls(session, sandbox)
        try:
            yield instance
        finally:
            try:
                await instance.aclose()
            finally:
                instance._closed = True
                instance._loaded.clear()
                instance._details.clear()
                await instance._stack.aclose()

    def register_tools(self, factory: ToolFactory) -> None:
        """注册 Notebook 和类发现工具，目录仅展示类级文档。

        Args:
            factory: 当前会话启动期间的工具注册表。
        """
        factory.register_callable(
            "python__describe_class",
            self.describe_class,
            description=(
                "以 Python 类文档形式查看全部公开方法的签名、类型注解和完整 docstring，"
                "并在当前会话加载该类。使用业务方法前必须先调用此工具。"
            ),
        )
        super().register_tools(factory)
        catalog = [
            "# Python 工具类",
            (
                "以下仅列出类级功能概览。先通过 python__describe_class 查看所需类的完整接口，"
                "再在 Notebook cell 中通过 ClassName.method(...) 同步调用；未查看的类不可调用。"
                "所有代码验证都必须写入 Notebook，再用 notebook__exec_cell 执行。"
                "同一 Notebook 的变量跨 cell 保留；需要独立代码片段或全新变量空间时，"
                "使用 notebook__create_notebook 创建新的 Notebook。不要猜测参数或返回字段。"
            ),
        ]
        for name, tool_class in self._classes.items():
            catalog.append(f"## {name}\n{inspect.getdoc(tool_class)}")
        factory.add_instructions("\n\n".join(catalog))

    async def describe_class(self, class_name: str) -> str:
        """加载一个工具类，以 Python 文档形式返回全部公开方法的接口。

        Args:
            class_name: 初始类级目录中的精确类名。

        Returns:
            包含类级文档、全部公开方法签名及完整 docstring 的 Python 代码块。
            保留实际换行，不附带 JSON Schema；重复查看不重复创建实例。

        Raises:
            ValueError: 类名未知，或公开方法清单、文档、类型声明不合法。
            Exception: 资源绑定或 Schema 构建失败；不会开放部分加载的类。
        """
        self._require_open()
        if class_name in self._details:
            return self._details[class_name]
        if class_name not in self._classes:
            raise ValueError(f"未知的 Python 工具类：{class_name}")
        tool_class = self._classes[class_name]
        async with AsyncExitStack() as pending:
            instance = await pending.enter_async_context(tool_class.bind(self._session))
            if not isinstance(instance, tool_class):
                raise TypeError(f"{class_name}.bind 没有返回匹配的工具实例。")
            methods = instance.methods()
            if not methods or any(
                not name.isidentifier() or name.startswith("_") or iskeyword(name)
                for name in methods
            ):
                raise ValueError("工具必须显式提供合法的公开方法名。")
            bindings = {name: ToolMethod(name, function) for name, function in methods.items()}
            details = ClassDetails(
                class_name=class_name,
                docstring=inspect.getdoc(tool_class) or "",
                methods=[binding.details for binding in bindings.values()],
            )
            documentation = render_class_documentation(details)
            await self._stack.enter_async_context(pending.pop_all())
        self._loaded[class_name] = bindings
        self._details[class_name] = documentation
        return documentation

    async def _call_method(self, payload: dict[str, Any]) -> Any:
        """只调度当前会话已查看详情且显式公开的方法。"""
        self._require_open()
        request = PythonToolCall.model_validate(payload)
        if request.class_name not in self._loaded:
            raise PermissionError("必须先通过 python__describe_class 查看该类的完整接口。")
        methods = self._loaded[request.class_name]
        if request.method_name not in methods:
            raise PermissionError("该方法不在工具类的公开接口中。")
        return await methods[request.method_name].call(request.args, request.kwargs)

    def _require_open(self) -> None:
        """拒绝通过宿主保留的旧引用重新加载或调用已关闭的工具。"""
        if self._closed:
            raise RuntimeError("Python 工具会话已经关闭。")

    def _create_kernel(self) -> NotebookKernel:
        """为每个 Notebook 配置同一会话授权目录的受限调用桥接。"""
        return NotebookKernel(
            self.sandbox, request_handler=self._call_method, bootstrap=self._proxy_source
        )

    def _proxy_source(self) -> str:
        """在单元执行前同步已加载类，不改写文档或清除 Notebook 变量。"""
        self._require_open()
        classes = {name: list(methods) for name, methods in self._loaded.items()}
        worker = Path(__file__).with_name("worker.py").read_text(encoding="utf-8")
        source = f"{worker}\ninstall({classes!r}, namespace)\n"
        return f"exec({source!r}, {{'namespace': get_ipython().user_ns}})"
