"""在可信引导阶段安装 Landlock，再用 exec 替换为目标 Shell。"""

import json
import sys
from pathlib import Path

from ..config import SandboxConfig
from .bubblewrap import build_bubblewrap_command

# 引导源码由宿主提供，用户命令只作为 JSON 参数传递，不拼接进 Python 源码。
_BOOTSTRAP = """
import json, os, sys
from nooa.runtime.sandbox.config import LandlockRule
from nooa.runtime.sandbox.guards import apply_landlock, landlock_abi
request = json.loads(sys.argv[1])
if landlock_abi() < 3:
    raise RuntimeError('Landlock ABI >= 3 is required to restrict file truncation')
rules = [LandlockRule(path, required=True) for path in request['reads']]
rules.extend([
    LandlockRule(request['workdir'], write=True, required=True),
    LandlockRule('/dev/null', write=True, required=True),
    LandlockRule('/dev/random', required=True),
    LandlockRule('/dev/urandom', required=True),
])
if request['ipc_dir'] is not None:
    rules.append(LandlockRule(request['ipc_dir'], write=True, required=True))
apply_landlock(rules)
os.execv(request['argv'][0], request['argv'])
"""


def build_landlock_command(
    config: SandboxConfig, argv: list[str], reads: tuple[Path, ...], *, ipc_dir: Path | None = None
) -> list[str]:
    """复用 NOOA 的 Landlock 原语，配合私有 PID、挂载和网络空间执行。

    Args:
        config: 唯一可写工作目录和网络权限。
        argv: 安装限制后才执行的目标程序与参数。
        reads: Landlock 允许读取的依赖与输入。
        ipc_dir: 宿主创建的专属通信目录，挂载与 Landlock 使用同一范围。

    Returns:
        先隔离命名空间、再安装文件权限、最后 exec 目标程序的命令。

    Raises:
        RuntimeError: bubblewrap 不可用；Landlock 安装失败则子进程非零退出。
    """
    payload = json.dumps(
        {
            "reads": [str(path) for path in reads],
            "workdir": str(config.workdir),
            "argv": argv,
            "ipc_dir": str(ipc_dir) if ipc_dir is not None else None,
        }
    )
    return build_bubblewrap_command(
        config, [sys.executable, "-I", "-c", _BOOTSTRAP, payload], reads, ipc_dir=ipc_dir
    )
