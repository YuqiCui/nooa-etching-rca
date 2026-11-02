"""使用 Linux 私有挂载、PID 和网络命名空间隔离命令。"""

import os
import shutil
import sys
from pathlib import Path

from ..config import SandboxConfig


def namespace_command(config: SandboxConfig) -> list[str]:
    """创建两个 Linux 后端共用的命名空间前缀。

    Args:
        config: 网络是否共享宿主的固定权限。

    Returns:
        尚未指定挂载和目标程序的 bubblewrap 参数。

    Raises:
        RuntimeError: 当前平台或安装条件不支持 bubblewrap。
    """
    binary = shutil.which("bwrap")
    if not sys.platform.startswith("linux") or binary is None:
        raise RuntimeError("Linux sandbox requires bubblewrap with usable user/PID namespaces")
    return [
        binary,
        "--unshare-all",
        "--die-with-parent",
        "--cap-drop",
        "ALL",
        *(["--share-net"] if config.allow_network else []),
    ]


def build_bubblewrap_command(
    config: SandboxConfig, argv: list[str], reads: tuple[Path, ...], *, ipc_dir: Path | None = None
) -> list[str]:
    """仅把授权目录挂载到私有根目录，然后直接执行目标程序。

    Args:
        config: 工作目录与网络权限。
        argv: 需要执行的程序及其原样参数。
        reads: 已规范化的只读路径。
        ipc_dir: 宿主创建的专属通信目录；不共享宿主网络空间。

    Returns:
        具有独立根目录和只读依赖的完整命令参数。

    Raises:
        RuntimeError: bubblewrap 不可用。
    """
    command = [*namespace_command(config), "--tmpfs", "/"]
    for alias in ("/bin", "/sbin", "/lib", "/lib64"):
        if Path(alias).is_symlink():
            command.extend(["--symlink", os.readlink(alias), alias])
    for path in sorted(reads, key=lambda value: (len(value.parts), str(value))):
        command.extend(["--ro-bind", str(path), str(path)])
    if ipc_dir is not None:
        command.extend(["--bind", str(ipc_dir), str(ipc_dir)])
    command.extend(
        [
            "--bind",
            str(config.workdir),
            str(config.workdir),
            "--proc",
            "/proc",
            "--remount-ro",
            "/proc",
            "--dev",
            "/dev",
            "--chdir",
            str(config.workdir),
            "--remount-ro",
            "/",
            "--",
            *argv,
        ]
    )
    return command
