"""选择明确的隔离后端；不提供无文件隔离的回退。"""

import sys
from pathlib import Path
from typing import Literal

from ..config import SandboxConfig
from .bubblewrap import build_bubblewrap_command
from .landlock import build_landlock_command
from .macos import build_macos_command


def build_command(
    config: SandboxConfig, argv: list[str], reads: tuple[Path, ...], *, ipc_dir: Path | None = None
) -> tuple[Literal["macos", "bubblewrap", "landlock"], list[str]]:
    """按宿主选择或显式配置生成执行命令。

    Args:
        config: 宿主绑定的权限配置。
        argv: 未经 Shell 二次解析的程序参数。
        reads: 完整只读授权范围。
        ipc_dir: 宿主创建的临时 Jupyter 通信目录；None 不增加本地通信权限。

    Returns:
        实际后端名称与执行参数列表。

    Raises:
        RuntimeError: 平台或后端不可用；不会改用无隔离执行。
    """
    backend = config.backend
    if backend == "auto":
        if sys.platform == "darwin":
            backend = "macos"
        elif sys.platform.startswith("linux"):
            backend = "bubblewrap"
        else:
            raise RuntimeError(f"Unsupported sandbox platform: {sys.platform}")
    if backend == "macos":
        return backend, build_macos_command(config, argv, reads, ipc_dir=ipc_dir)
    if backend == "bubblewrap":
        return backend, build_bubblewrap_command(config, argv, reads, ipc_dir=ipc_dir)
    return "landlock", build_landlock_command(config, argv, reads, ipc_dir=ipc_dir)
