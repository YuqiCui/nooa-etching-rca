"""解析只读运行依赖，构造不继承宿主凭据的环境。"""

import sys
from pathlib import Path


def runtime_read_paths() -> tuple[Path, ...]:
    """收集当前解释器、系统库和科学计算字体需要读取的已有路径。"""
    candidates = (
        sys.prefix,
        sys.base_prefix,
        "/usr",
        "/bin",
        "/sbin",
        "/lib",
        "/lib64",
        "/System",
        "/Library/Apple",
        "/Library/Fonts",
        "/etc/ld.so.cache",
        "/etc/localtime",
        "/etc/fonts",
        "/var/cache/fontconfig",
        "/private/var/db/timezone",
    )
    return tuple(
        dict.fromkeys(Path(value).resolve() for value in candidates if Path(value).exists())
    )


def network_read_paths() -> tuple[Path, ...]:
    """收集联网时需要的解析器和证书配置，不授权整个 /etc。"""
    return tuple(
        Path(value).resolve()
        for value in (
            "/etc/resolv.conf",
            "/etc/hosts",
            "/etc/nsswitch.conf",
            "/etc/ssl",
            "/etc/pki",
        )
        if Path(value).exists()
    )


def command_environment(workdir: Path) -> dict[str, str]:
    """重新构造 Shell 环境；不继承凭据、启动脚本和动态链接器配置。

    Args:
        workdir: 固定的工作目录，同时承载临时文件和用户缓存。

    Returns:
        传给子进程的最小环境变量集合。
    """
    return {
        "PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(workdir),
        "TMPDIR": str(workdir),
        "XDG_CACHE_HOME": str(workdir),
        "MPLCONFIGDIR": str(workdir),
        "MPLBACKEND": "Agg",
        "PYTHON_DOTENV_DISABLED": "1",
        "LITELLM_MODE": "PRODUCTION",
        "PYTHONUNBUFFERED": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "POLARS_MAX_THREADS": "2",
    }
