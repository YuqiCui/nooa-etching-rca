"""通过 macOS 系统策略直接隔离 Shell 及其子进程。"""

import json
import sys
from pathlib import Path

from ..config import SandboxConfig


def build_macos_command(
    config: SandboxConfig, argv: list[str], reads: tuple[Path, ...], *, ipc_dir: Path | None = None
) -> list[str]:
    """生成默认拒绝的 sandbox-exec 命令。

    Args:
        config: 固定工作目录和网络权限。
        argv: 需要执行的程序及其原样参数。
        reads: 已解析的全部只读依赖和输入。
        ipc_dir: 宿主创建的专属通信目录，仅开放该目录中的 Unix socket。

    Returns:
        可直接传给 create_subprocess_exec 的参数列表。

    Raises:
        RuntimeError: 当前平台不支持 sandbox-exec。
    """
    binary = Path("/usr/bin/sandbox-exec")
    if sys.platform != "darwin" or not binary.is_file():
        raise RuntimeError("macos backend requires /usr/bin/sandbox-exec on macOS")
    read_rules = " ".join(f"(subpath {json.dumps(str(path))})" for path in reads)
    work_rule = f"(subpath {json.dumps(str(config.workdir))})"
    profile = (
        "(version 1)(deny default)"
        "(allow process-exec)(allow process-fork)(allow sysctl-read)"
        "(allow process-info* (target self))"
        "(allow file-read-metadata)"
        f"(allow file-read* {read_rules} {work_rule})"
        f'(allow file-write* {work_rule} (literal "/dev/null"))'
        '(allow file-read-data (literal "/dev/null") (literal "/dev/random")'
        ' (literal "/dev/urandom") (literal "/"))'
        "(allow ipc-posix-shm)(allow ipc-posix-sem)"
        '(allow mach-lookup (global-name "com.apple.cfprefsd.agent")'
        ' (global-name "com.apple.cfprefsd.daemon")'
        ' (global-name "com.apple.FontObjectsServer"))'
        + ("(allow network*)" if config.allow_network else "")
    )
    if ipc_dir is not None:
        ipc_rule = f"(subpath {json.dumps(str(ipc_dir))})"
        profile += (
            f"(allow file-read* file-write* {ipc_rule})"
            "(allow system-socket (socket-domain AF_UNIX))"
            f"(allow network-bind (local unix-socket {ipc_rule}))"
            f"(allow network-inbound (local unix-socket {ipc_rule}))"
            f"(allow network-outbound (remote unix-socket {ipc_rule}))"
        )
    return [str(binary), "-p", profile, *argv]
