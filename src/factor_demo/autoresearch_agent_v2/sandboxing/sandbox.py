"""Shell 和 Python 共用的工作空间隔离入口。"""

import asyncio
import math
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from .backends import build_command
from .bridge import ToolRequestHandler
from .config import SandboxConfig
from .execution import execute_command
from .paths import network_read_paths, runtime_read_paths
from .typing import SandboxResult


class Sandbox:
    """绑定一个固定工作空间，每次调用启动独立的受限 Shell。"""

    def __init__(self, config: SandboxConfig) -> None:
        """保存不可变权限并解析运行依赖，不修改 Session 状态。

        Args:
            config: 由宿主按 Session 工作目录创建的权限配置。

        Raises:
            ValueError: 工作目录覆盖运行依赖，或只读输入位于可写目录内。
        """
        self._config = config
        self._closed = False
        self._running: set[asyncio.Task] = set()
        runtime = runtime_read_paths()
        if any(path.is_relative_to(config.workdir) for path in runtime):
            raise ValueError("workdir cannot contain the interpreter or system runtime")
        if any(path.is_relative_to(config.workdir) for path in config.read_paths):
            raise ValueError("read_paths must be outside the writable workdir")
        self._reads = tuple(
            dict.fromkeys(
                (
                    *runtime,
                    *config.read_paths,
                    *(network_read_paths() if config.allow_network else ()),
                )
            )
        )

    @property
    def config(self) -> SandboxConfig:
        """返回本执行环境绑定的不可变权限。"""
        return self._config

    def prepare_command(
        self, argv: list[str], *, ipc_dir: Path | None = None
    ) -> tuple[Literal["macos", "bubblewrap", "landlock"], list[str]]:
        """为宿主持有的常驻进程生成相同权限的隔离命令。

        Args:
            argv: 直接执行的程序参数，不经过 Shell 解析。
            ipc_dir: 宿主创建并管理的临时通信目录，不能由模型提供。

        Returns:
            实际后端及命令；进程生命周期由调用资源负责。

        Raises:
            RuntimeError: 沙箱已关闭或后端不可用。
        """
        if self._closed:
            raise RuntimeError("Sandbox is closed")
        return build_command(self._config, argv, self._reads, ipc_dir=ipc_dir)

    async def aclose(self) -> None:
        """撤销后续执行权限，取消并等待已经开始的执行完成清理。"""
        self._closed = True
        running = tuple(self._running)
        for task in running:
            if not task.cancelling():
                task.cancel()
        if running:
            await asyncio.gather(*running, return_exceptions=True)

    async def run_shell(
        self,
        command: str,
        *,
        timeout_s: float | None = None,
        request_handler: ToolRequestHandler | None = None,
    ) -> SandboxResult:
        """在隔离环境中执行非登录、非交互的 /bin/sh 命令。

        Args:
            command: Shell 源码，支持管道、重定向和多行命令。
            timeout_s: 可缩短本次超时，不能超过配置上限；None 使用配置值。
            request_handler: 宿主工具桥接入口；普通 Shell 调用保持为 None。

        Returns:
            本次命令的退出状态和有界输出；每次执行不保留 Shell 变量。

        Raises:
            ValueError: 超时超出范围或命令包含 NUL 字符。
            RuntimeError: 沙箱已关闭或无可用隔离后端，不会回退为普通执行。
            OSError: 后端进程无法启动或清理。
            asyncio.CancelledError: 调用取消，清理后原样传播。
        """
        if self._closed:
            raise RuntimeError("Sandbox is closed")
        timeout = self._config.timeout_s if timeout_s is None else timeout_s
        if not math.isfinite(timeout) or not 0 < timeout <= self._config.timeout_s:
            raise ValueError("timeout_s must be positive and cannot exceed the configured limit")
        if "\0" in command:
            raise ValueError("command cannot contain NUL characters")
        backend, argv = build_command(self._config, ["/bin/sh", "-c", command], self._reads)
        # 单独跟踪执行任务，关闭时不取消借用资源的宿主协程。
        running = asyncio.create_task(
            execute_command(self._config, backend, argv, timeout, request_handler)
        )
        self._running.add(running)
        try:
            return await running
        finally:
            self._running.discard(running)

    async def run_python(
        self,
        code: str,
        *,
        args: Sequence[str] = (),
        timeout_s: float | None = None,
        request_handler: ToolRequestHandler | None = None,
    ) -> SandboxResult:
        """把 Python 源码及参数引用为 Shell 命令，复用 run_shell。

        Args:
            code: 本次执行的 Python 源码，不保留前次调用的内存变量。
            args: 传入 sys.argv[1:] 的原样字符串参数。
            timeout_s: 本次墙钟预算，约束与 run_shell 相同。
            request_handler: 可选的宿主工具桥接入口，不会放入子进程的 Python 对象中。

        Returns:
            与 Shell 入口完全相同的执行结果。

        Raises:
            ValueError: 超时非法或参数包含 NUL 字符。
            RuntimeError: 沙箱已关闭或无可用隔离后端。
            OSError: 进程启动或清理失败。
            asyncio.CancelledError: 调用方取消执行。
        """
        command = shlex.join([sys.executable, "-I", "-c", code, *args])
        if request_handler is not None:
            return await self.run_shell(
                f"exec {command}", timeout_s=timeout_s, request_handler=request_handler
            )
        return await self.run_shell(f"exec {command}", timeout_s=timeout_s)
