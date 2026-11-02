"""在现有隔离边界内启动真实 Jupyter kernel，并通过本地 IPC 通信。"""

import asyncio
import json
import os
import secrets
import signal
import sys
from collections.abc import Callable
from pathlib import Path
from queue import Empty
from tempfile import TemporaryDirectory
from typing import Any

from jupyter_client import AsyncKernelClient
from jupyter_client.connect import write_connection_file
from nbformat.v4 import output_from_msg

from ....sandboxing import Sandbox
from ....sandboxing.bridge import BRIDGE_PREFIX, MAX_FRAME_BYTES, ToolRequestHandler
from ....sandboxing.paths import command_environment
from .typing import CellExecution


class NotebookKernel:
    """持有一个 Notebook 的进程与客户端，直到会话结束或执行失败。"""

    def __init__(
        self,
        sandbox: Sandbox,
        *,
        request_handler: ToolRequestHandler | None = None,
        bootstrap: Callable[[], str] | None = None,
    ) -> None:
        """保存宿主授予的权限，构造时不启动进程。

        Args:
            sandbox: 当前 Session 共享的执行权限资源。
            request_handler: 仅处理当前会话已授权类方法的宿主入口。
            bootstrap: 单元执行前同步代理的宿主源码工厂，不接收模型源码。
        """
        self.sandbox = sandbox
        self._request_handler = request_handler
        self._bootstrap = bootstrap
        self.process: asyncio.subprocess.Process | None = None
        self.client: AsyncKernelClient | None = None
        self._temporary: TemporaryDirectory | None = None
        self._drains: list[asyncio.Task] = []
        self._diagnostics = bytearray()

    @property
    def connected(self) -> bool:
        """仅在客户端已连接且进程仍存活时返回 True。"""
        return (
            self.client is not None and self.process is not None and self.process.returncode is None
        )

    async def _drain_diagnostics(self, reader: asyncio.StreamReader) -> None:
        """排空进程日志，保留有界尾部用于报告启动失败。"""
        while chunk := await reader.read(65536):
            self._diagnostics.extend(chunk)
            del self._diagnostics[:-4000]

    async def start(self) -> None:
        """启动并等待 kernel 就绪，不读取宿主的 Jupyter 配置。

        Raises:
            RuntimeError: 隔离后端或 kernel 启动失败，资源已清理。
            asyncio.CancelledError: 启动被取消，清理后继续传播。
        """
        try:
            # 使用短路径避免 Unix socket 地址长度限制；目录只承载本 kernel 的通信。
            self._temporary = TemporaryDirectory(prefix="nk-", dir="/tmp")
            runtime = Path(self._temporary.name).resolve()
            connection = runtime / "connection.json"
            write_connection_file(
                str(connection),
                transport="ipc",
                ip=str(runtime / "k"),
                shell_port=1,
                iopub_port=2,
                stdin_port=3,
                hb_port=4,
                control_port=5,
                key=secrets.token_hex(32).encode("ascii"),
            )
            _, command = self.sandbox.prepare_command(
                [
                    sys.executable,
                    "-I",
                    "-m",
                    "ipykernel_launcher",
                    "-f",
                    str(connection),
                    "--HistoryManager.enabled=False",
                    "--IPKernelApp.capture_fd_output=False",
                    "--IPKernelApp.matplotlib=inline",
                ],
                ipc_dir=runtime,
            )
            self.process = await asyncio.create_subprocess_exec(
                *command,
                cwd=self.sandbox.config.workdir,
                env=command_environment(self.sandbox.config.workdir),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
                close_fds=True,
            )
            for reader in (self.process.stdout, self.process.stderr):
                assert reader is not None
                self._drains.append(asyncio.create_task(self._drain_diagnostics(reader)))
            self.client = AsyncKernelClient(connection_file=str(connection))
            self.client.load_connection_file()
            self.client.start_channels()
            await self.client.wait_for_ready(timeout=20)
        except BaseException as exc:
            await self.aclose()
            if isinstance(exc, asyncio.CancelledError):
                raise
            detail = self._diagnostics.decode("utf-8", errors="replace")
            raise RuntimeError(f"Jupyter kernel failed to start: {detail or str(exc)}") from exc

    async def aclose(self) -> None:
        """关闭通道、回收进程组和临时通信目录，重复调用安全。"""
        if self.client is not None:
            self.client.stop_channels()
            self.client = None
        if self.process is not None:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await self.process.wait()
            self.process = None
        await asyncio.gather(*self._drains, return_exceptions=True)
        self._drains.clear()
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None

    async def execute(self, source: str, timeout_s: float) -> CellExecution:
        """发送源码，按请求标识收集输出，直到收到 idle 和执行回复。

        Args:
            source: 此次明确指定版本的 Python 源码。
            timeout_s: 整次执行的墙钟预算，单位秒。

        Returns:
            有界 nbformat 输出与执行状态。普通 Python 异常保留 kernel 内存；
            超时或崩溃会关闭当前 kernel，不影响其他 Notebook。

        Raises:
            RuntimeError: kernel 已断开，不自动创建新进程掩盖内存丢失。
            asyncio.CancelledError: 执行被取消，清理当前 kernel 后传播。
        """
        if not self.connected:
            raise RuntimeError("Notebook kernel is disconnected")
        result = CellExecution(status="completed")
        try:
            async with asyncio.timeout(timeout_s):
                if self._bootstrap is not None:
                    initialized = await self._execute_request(self._bootstrap(), silent=True)
                    result = initialized
                if result.status == "completed":
                    await self._execute_request(source, result=result)
        except TimeoutError:
            result.status = "timeout" if self.connected else "crashed"
        except asyncio.CancelledError:
            await self.aclose()
            raise
        except Exception:
            await self.aclose()
            raise
        if result.status in {"timeout", "crashed"}:
            await self.aclose()
        return result

    async def _serve_tool_requests(self, request: str) -> None:
        """只处理当前执行的 stdin 工具请求，拒绝交互输入和未授权调用。"""
        assert self.client is not None
        client = self.client
        while True:
            message = await client.get_stdin_msg()
            if message["parent_header"].get("msg_id") != request:
                continue
            try:
                prompt = message["content"]["prompt"]
                prefix = BRIDGE_PREFIX.decode("ascii")
                if self._request_handler is None or not prompt.startswith(prefix):
                    raise PermissionError("Notebook 不支持交互输入，只允许已授权的工具调用。")
                payload = prompt[len(prefix) :]
                if len(payload.encode("utf-8")) > MAX_FRAME_BYTES:
                    raise ValueError("单次工具请求超过 1 MiB，请缩小参数。")
                result = await self._request_handler(json.loads(payload))
                response = json.dumps({"result": result}, ensure_ascii=False, allow_nan=False)
                if len(response.encode("utf-8")) > MAX_FRAME_BYTES:
                    raise ValueError("工具返回值超过 1 MiB，请缩小查询范围或返回产物引用。")
            except Exception as exc:  # noqa: BLE001 -- 将业务异常回传给单元代码处理。
                response = json.dumps(
                    {"error": {"type": type(exc).__name__, "message": str(exc)[:4096]}},
                    ensure_ascii=False,
                )
            client.input(response)

    async def _execute_request(
        self, source: str, *, silent: bool = False, result: CellExecution | None = None
    ) -> CellExecution:
        """收集一次 Jupyter 请求，桥接任务与该请求同生共灭。"""
        assert self.client is not None
        client = self.client
        if result is None:
            result = CellExecution(status="completed")
        used_bytes = 2
        clear_next = False
        displays: dict[str, list[dict[str, Any]]] = {}
        request = client.execute(
            source,
            silent=silent,
            store_history=not silent,
            allow_stdin=self._request_handler is not None,
            stop_on_error=True,
        )
        bridge = asyncio.create_task(self._serve_tool_requests(request))
        try:
            while True:
                try:
                    message = await client.get_iopub_msg(timeout=0.5)
                except Empty:
                    if not self.connected:
                        result.status = "crashed"
                        break
                    continue
                if message["parent_header"].get("msg_id") != request:
                    continue
                kind, content = message["msg_type"], message["content"]
                if kind == "status" and content["execution_state"] == "idle":
                    break
                if kind == "execute_input":
                    result.execution_count = content["execution_count"]
                elif kind == "clear_output":
                    clear_next = content.get("wait", False)
                    if not clear_next:
                        result.outputs.clear()
                        displays.clear()
                        used_bytes = 2
                elif kind in {
                    "stream",
                    "execute_result",
                    "display_data",
                    "error",
                    "update_display_data",
                }:
                    if kind == "error":
                        result.status = "error"
                    if clear_next:
                        result.outputs.clear()
                        displays.clear()
                        used_bytes, clear_next = 2, False
                    display_id = content.get("transient", {}).get("display_id")
                    if kind == "update_display_data":
                        previous = displays.get(display_id, [])
                        proposed = [
                            {**output, "data": content["data"], "metadata": content["metadata"]}
                            for output in previous
                        ]
                    else:
                        proposed = [dict(output_from_msg(message))]
                    # display_id 可对应多份输出，更新时必须计算每一份的大小。
                    size = sum(
                        len(json.dumps(output, ensure_ascii=False).encode("utf-8")) + 2
                        for output in proposed
                    )
                    if used_bytes + size > self.sandbox.config.max_output_bytes:
                        result.output_truncated = True
                        continue
                    used_bytes += size
                    if kind == "update_display_data":
                        for output, updated in zip(previous, proposed, strict=True):
                            output.update(updated)
                        continue
                    output = proposed[0]
                    result.outputs.append(output)
                    if display_id:
                        displays.setdefault(display_id, []).append(output)
            if result.status != "crashed":
                while True:
                    reply = await client.get_shell_msg()
                    if reply["parent_header"].get("msg_id") == request:
                        result.execution_count = reply["content"]["execution_count"]
                        if reply["content"]["status"] != "ok":
                            result.status = "error"
                        break
        finally:
            bridge.cancel()
            await asyncio.gather(bridge, return_exceptions=True)
        return result
