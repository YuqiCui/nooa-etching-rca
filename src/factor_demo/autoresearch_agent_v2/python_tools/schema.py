"""由同一方法生成完整描述和严格的调用校验器。"""

import asyncio
import inspect
from collections.abc import Callable
from functools import wraps
from typing import Any, get_type_hints

from pydantic import ConfigDict, TypeAdapter
from pydantic_core import ArgsKwargs

from .typing import MethodDetails


class ToolMethod:
    """保存一个已公开方法的接口与校验器，不交给执行进程反射。"""

    def __init__(self, name: str, function: Callable[..., Any]) -> None:
        """检查文档和类型声明，生成与实际调用一致的描述。

        Args:
            name: 工具类显式公开的方法名。
            function: 当前会话实例的同步或异步绑定方法。

        Raises:
            ValueError: 方法缺少完整文档或参数、返回值类型注解。
            Exception: 类型不能生成 Schema 或校验器。
        """
        docstring = inspect.getdoc(function)
        signature = inspect.signature(function, eval_str=True)
        hints = get_type_hints(function, include_extras=True)
        if not docstring or "return" not in hints:
            raise ValueError(f"{name} 必须提供完整文档和返回类型。")
        if any(parameter not in hints for parameter in signature.parameters):
            raise ValueError(f"{name} 的所有参数都必须提供类型注解。")

        @wraps(function)
        def invoke(*args: Any, **kwargs: Any) -> Any:
            """保留绑定方法签名，由参数校验器调用原方法。"""
            return function(*args, **kwargs)

        self._function = function
        self._input = TypeAdapter(invoke, config=ConfigDict(extra="forbid", strict=True))
        self._output = TypeAdapter(hints["return"])
        self.details = MethodDetails(
            name=name,
            signature=f"{name}{signature}",
            docstring=docstring,
            input_schema=self._input.json_schema(),
            output_schema=self._output.json_schema(),
        )

    async def call(self, args: list[Any], kwargs: dict[str, Any]) -> Any:
        """严格校验参数、执行方法并校验和序列化返回值。

        Args:
            args: Python 位置参数。
            kwargs: Python 关键字参数。

        Returns:
            满足声明的返回类型、可通过 JSON 传回执行进程的值。

        Raises:
            pydantic.ValidationError: 参数或返回值不满足声明。
            Exception: 工具执行失败，保留原始错误。
        """
        arguments = ArgsKwargs(tuple(args), kwargs)
        if inspect.iscoroutinefunction(self._function):
            result = self._input.validate_python(arguments)
        else:
            running = asyncio.create_task(asyncio.to_thread(self._input.validate_python, arguments))
            try:
                result = await asyncio.shield(running)
            except asyncio.CancelledError:
                # 同步宿主调用不能强制中断，等待其退出后才能安全释放所借用的资源。
                await asyncio.gather(running, return_exceptions=True)
                raise
        if inspect.isawaitable(result):
            result = await result
        validated = self._output.validate_python(result, strict=True)
        return self._output.dump_python(validated, mode="json")
