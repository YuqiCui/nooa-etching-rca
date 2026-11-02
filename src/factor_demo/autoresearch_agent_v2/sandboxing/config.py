"""声明工作空间的执行权限和单次命令预算。"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SandboxConfig(BaseModel):
    """由宿主创建的固定权限；模型不能通过命令参数扩大权限。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    """拒绝未知配置，并禁止执行期间修改权限字段。"""

    workdir: Path
    """已存在的可读写工作目录；唯一的业务文件写入范围。"""

    read_paths: tuple[Path, ...] = ()
    """额外授权的只读文件或目录；不能放在可写工作目录内部。"""

    backend: Literal["auto", "macos", "bubblewrap", "landlock"] = "auto"
    """隔离后端；auto 按平台选择，失败时不降级为普通进程。"""

    allow_network: bool = False
    """是否允许网络；开启后不限制目标域名和地址。"""

    timeout_s: float = Field(default=120, gt=0, allow_inf_nan=False)
    """单次运行的墙钟预算，单位秒。"""

    max_output_bytes: int = Field(default=262144, gt=0)
    """stdout 和 stderr 各自保留的字节上限；超出后继续排空管道。"""

    @field_validator("workdir")
    @classmethod
    def validate_workdir(cls, value: Path) -> Path:
        """固定已存在的真实工作目录，拒绝授权整个文件系统。"""
        path = value.expanduser().resolve(strict=True)
        if not path.is_dir() or path == Path(path.anchor):
            raise ValueError("workdir must be an existing directory other than filesystem root")
        return path

    @field_validator("read_paths")
    @classmethod
    def validate_read_paths(cls, values: tuple[Path, ...]) -> tuple[Path, ...]:
        """规范化已有的只读路径，拒绝授权整个文件系统。"""
        paths = tuple(dict.fromkeys(path.expanduser().resolve(strict=True) for path in values))
        if any(path == Path(path.anchor) for path in paths):
            raise ValueError("read_paths cannot include filesystem root")
        return paths
