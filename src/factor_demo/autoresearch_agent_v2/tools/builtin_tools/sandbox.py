"""把隔离 Shell 绑定到 Session，支持文件操作和脚本执行。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import partial
from typing import TYPE_CHECKING, ClassVar

from ...sandboxing import Sandbox, SandboxConfig, SandboxResult
from ..base import ToolFactory

if TYPE_CHECKING:
    from ...agent.session import Session


class SandboxTools:
    """同一会话的执行工具共用一个具有固定权限的 Sandbox。"""

    namespace: ClassVar[str] = "sandbox"
    """Shell 工具使用的名称前缀。"""

    def __init__(self, sandbox: Sandbox) -> None:
        """绑定 Session 已经创建的执行资源。

        Args:
            sandbox: 当前会话拥有并负责关闭的 Sandbox。
        """
        self.sandbox = sandbox

    @classmethod
    def register(cls, session: Session, *, config: SandboxConfig | None = None) -> None:
        """登记沙箱资源和工具组；实际资源在 Session 启动时创建。

        Args:
            session: 尚未启动的会话，提供唯一的可写工作目录。
            config: 宿主配置；None 使用当前工作目录和默认限制。配置不能改变
                session.state.workdir 的目录范围。

        Raises:
            ValueError: 配置与 Session 的工作目录不一致或资源重复登记。
            RuntimeError: 会话已经启动或关闭。
        """
        bound_config = config or SandboxConfig(workdir=session.state.workdir)
        cls._check_workdir(session, bound_config)

        session.register_resource(Sandbox, partial(cls.create_sandbox, config=bound_config))
        session.register_stateful_tool(cls)

    @classmethod
    @asynccontextmanager
    async def create_sandbox(
        cls, session: Session, *, config: SandboxConfig | None = None
    ) -> AsyncIterator[Sandbox]:
        """创建执行资源，并在退出时撤销权限和关闭子进程。

        Args:
            session: 提供当前工作目录的会话。
            config: 宿主预先提供的配置；None 使用会话目录和默认限制。

        Yields:
            当前会话负责关闭的 Sandbox。
        """
        bound_config = config or SandboxConfig(workdir=session.state.workdir)
        cls._check_workdir(session, bound_config)
        sandbox = Sandbox(bound_config)
        try:
            yield sandbox
        finally:
            await sandbox.aclose()

    @staticmethod
    def _check_workdir(session: Session, config: SandboxConfig) -> None:
        """拒绝不同工作目录以及登记后被宿主修改的会话范围。"""
        if config.workdir != session.state.workdir.expanduser().resolve(strict=True):
            raise ValueError("Sandbox workdir must match Session workdir")

    @classmethod
    @asynccontextmanager
    async def bind(cls, session: Session) -> AsyncIterator[SandboxTools]:
        """取得当前会话唯一的 Sandbox，创建共享该资源的工具组。

        Args:
            session: 已按顺序创建 Sandbox 资源的会话。

        Yields:
            Shell 方法绑定当前 Sandbox 的工具组。
        """
        sandbox = await session.bind_resource(Sandbox, cls.create_sandbox)
        cls._check_workdir(session, sandbox.config)
        yield cls(sandbox)

    def register_tools(self, factory: ToolFactory) -> None:
        """注册带参数校验的 Shell 工具。

        Args:
            factory: 当前会话启动期间使用的工具表。
        """
        factory.register_callable(
            f"{self.namespace}__run_shell",
            self.run_shell,
            description="在当前会话工作目录的沙箱内执行 Shell；支持管道和重定向。",
        )
        factory.add_instructions(
            f"当前会话单次执行的超时上限为 {self.sandbox.config.timeout_s:g} 秒；"
            "timeout_s 只能缩短该上限，省略时使用默认上限。"
            "Shell 用于工作目录内的文件操作和 Python 脚本执行。"
        )

    async def run_shell(self, command: str, timeout_s: float | None = None) -> SandboxResult:
        """在当前会话的固定目录和权限下运行 Shell。

        Args:
            command: 非交互 /bin/sh 源码，不能改变沙箱授权范围。
            timeout_s: 本次超时秒数，只能缩短配置预算；None 使用默认预算。

        Returns:
            可序列化的状态、退出码、有界 stdout 和 stderr。
        """
        return await self.sandbox.run_shell(command, timeout_s=timeout_s)
