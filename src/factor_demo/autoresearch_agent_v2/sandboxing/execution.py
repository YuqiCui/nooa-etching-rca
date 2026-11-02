"""执行后端命令，限制输出并在结束、超时或取消时清理进程组。"""

import asyncio
import os
import signal
import subprocess
import time
from typing import Literal

from .bridge import MAX_FRAME_BYTES, ToolRequestHandler, capture_tool_output
from .config import SandboxConfig
from .paths import command_environment
from .typing import SandboxResult


async def _capture_output(reader: asyncio.StreamReader, output: bytearray, limit: int) -> None:
    """持续排空输出，多保留一个字节用于在取消后仍可判断是否截断。"""
    while chunk := await reader.read(65536):
        available = max(0, limit + 1 - len(output))
        output.extend(chunk[:available])


async def _stop_process_group(process: subprocess.Popen) -> None:
    """终止整个进程组并等待主进程退出；Linux 额外由 PID 空间回收后代。"""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    await asyncio.to_thread(process.wait)


def _spawn_command(
    config: SandboxConfig, argv: list[str], with_tool_bridge: bool = False
) -> subprocess.Popen:
    """在独立进程组中启动后端，关闭宿主描述符并清空继承环境。"""
    return subprocess.Popen(
        argv,
        cwd=config.workdir,
        env=command_environment(config.workdir),
        stdin=subprocess.PIPE if with_tool_bridge else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        close_fds=True,
    )


async def execute_command(
    config: SandboxConfig,
    backend: Literal["macos", "bubblewrap", "landlock"],
    argv: list[str],
    timeout_s: float,
    request_handler: ToolRequestHandler | None = None,
) -> SandboxResult:
    """执行已经包装好隔离策略的命令。

    Args:
        config: 工作目录和输出保留预算。
        backend: 用于结果记录的实际后端。
        argv: 后端生成的完整命令，不使用宿主 Shell 再次解析。
        timeout_s: 本次墙钟预算，单位秒。
        request_handler: 可选的宿主工具调用入口；None 保持普通命令的标准管道行为。

    Returns:
        退出码、有界输出、超时标记和实际耗时。

    Raises:
        OSError: 程序启动或清理失败。
        asyncio.CancelledError: 调用方取消；先清理执行进程再传播取消。
    """
    started = time.monotonic()
    spawn = asyncio.create_task(
        asyncio.to_thread(_spawn_command, config, argv, request_handler is not None)
    )
    # 即使启动期间取消，也先取得句柄并回收子进程。
    try:
        process = await asyncio.shield(spawn)
    except asyncio.CancelledError:
        process = await spawn
        await _stop_process_group(process)
        for pipe in (process.stdin, process.stdout, process.stderr):
            if pipe is not None:
                pipe.close()
        raise
    stdout, stderr = bytearray(), bytearray()
    transports = []
    captures = []
    timed_out = False
    try:
        async with asyncio.timeout(max(0, timeout_s - (time.monotonic() - started))):
            loop = asyncio.get_running_loop()
            writer = None
            if request_handler is not None:
                assert process.stdin is not None
                protocol = asyncio.streams.FlowControlMixin(loop=loop)
                transport, _ = await loop.connect_write_pipe(lambda: protocol, process.stdin)
                transports.append(transport)
                writer = asyncio.StreamWriter(transport, protocol, None, loop)
            for pipe, output in ((process.stdout, stdout), (process.stderr, stderr)):
                assert pipe is not None
                reader = asyncio.StreamReader(limit=MAX_FRAME_BYTES)
                transport, _ = await loop.connect_read_pipe(
                    lambda reader=reader: asyncio.StreamReaderProtocol(reader), pipe
                )
                transports.append(transport)
                if pipe is process.stdout and request_handler is not None:
                    assert writer is not None
                    capture = capture_tool_output(
                        reader, writer, output, config.max_output_bytes, request_handler
                    )
                else:
                    capture = _capture_output(reader, output, config.max_output_bytes)
                captures.append(asyncio.create_task(capture))
            await asyncio.gather(asyncio.to_thread(process.wait), *captures)
    except TimeoutError:
        timed_out = True
    finally:
        # 正常退出也清理同组后台任务；先发出终止信号，再排空剩余输出。
        cleanup = asyncio.create_task(_stop_process_group(process))
        try:
            await asyncio.shield(cleanup)
        finally:
            await cleanup
            for transport in transports:
                transport.close()
            for pipe in (process.stdin, process.stdout, process.stderr):
                if pipe is not None:
                    pipe.close()
            for capture in captures:
                if not capture.done():
                    capture.cancel()
            await asyncio.gather(*captures, return_exceptions=True)
    assert process.returncode is not None
    return SandboxResult(
        backend=backend,
        returncode=process.returncode,
        stdout=stdout[: config.max_output_bytes].decode("utf-8", errors="replace"),
        stderr=stderr[: config.max_output_bytes].decode("utf-8", errors="replace"),
        timed_out=timed_out,
        output_truncated=any(len(output) > config.max_output_bytes for output in (stdout, stderr)),
        duration_s=time.monotonic() - started,
    )
