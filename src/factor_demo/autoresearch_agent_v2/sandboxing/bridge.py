"""通过标准管道传递 Python 工具请求，不开放网络或继承宿主对象。"""

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

ToolRequestHandler = Callable[[dict[str, Any]], Awaitable[Any]]
"""由宿主校验并执行一个工具请求的异步入口。"""

BRIDGE_PREFIX = b"\x1ePYTHON_TOOL:"
"""区分桥接帧与扩展模块直接写出的普通标准输出。"""

MAX_FRAME_BYTES = 1024 * 1024
"""单条工具请求和响应的编码上限，避免无界缓冲。"""


class OutputFrame(BaseModel):
    """执行进程提交的一段普通标准输出。"""

    model_config = ConfigDict(extra="forbid", strict=True)
    """拒绝未知字段和隐式类型转换。"""

    kind: Literal["stdout"]
    """标识普通输出帧。"""

    text: str
    """待写入有界 stdout 预览的原始文本。"""


class CallFrame(BaseModel):
    """执行进程请求宿主调用已授权工具。"""

    model_config = ConfigDict(extra="forbid", strict=True)
    """拒绝未知字段和隐式类型转换。"""

    kind: Literal["call"]
    """标识工具调用帧。"""

    payload: dict[str, Any]
    """交给所属工具目录进一步校验的调用参数。"""


_FRAME_ADAPTER = TypeAdapter(Annotated[OutputFrame | CallFrame, Field(discriminator="kind")])
"""根据帧类型校验不可信的进程输入。"""


async def capture_tool_output(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    output: bytearray,
    limit: int,
    handler: ToolRequestHandler,
) -> None:
    """排空输出，并顺序处理经过校验的桥接帧。

    Args:
        reader: 执行进程的 stdout 管道。
        writer: 向执行进程 stdin 回传结果的管道。
        output: 仅保存普通输出的预览缓冲。
        limit: 预览字节上限，额外保留一字节判断截断。
        handler: 当前会话的工具调用入口，不接受进程自行提供的执行目标。

    Raises:
        ValueError: 帧超过限制或协议内容非法。
        OSError: 进程通信失败。
        asyncio.CancelledError: 执行被取消。
    """
    while line := await reader.readline():
        marker = line.find(BRIDGE_PREFIX)
        if marker < 0:
            output.extend(line[: max(0, limit + 1 - len(output))])
            continue
        output.extend(line[:marker][: max(0, limit + 1 - len(output))])
        frame = _FRAME_ADAPTER.validate_json(line[marker + len(BRIDGE_PREFIX) :])
        if isinstance(frame, OutputFrame):
            encoded = frame.text.encode("utf-8")
            output.extend(encoded[: max(0, limit + 1 - len(output))])
            continue
        try:
            result = await handler(frame.payload)
            response = json.dumps({"result": result}, ensure_ascii=False, allow_nan=False)
            if len(response.encode("utf-8")) > MAX_FRAME_BYTES:
                raise ValueError("工具返回值超过 1 MiB，请缩小查询范围或返回产物引用。")
        except Exception as exc:  # noqa: BLE001 -- 将业务工具异常回传给执行代码处理。
            response = json.dumps(
                {"error": {"type": type(exc).__name__, "message": str(exc)[:4096]}},
                ensure_ascii=False,
            )
        writer.write(response.encode("utf-8") + b"\n")
        await writer.drain()
