"""统一的命令执行结果。"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, computed_field


class SandboxResult(BaseModel):
    """保留实际退出状态和有界输出，不把命令失败转换成成功。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    """执行结果只读，拒绝未声明的字段。"""

    backend: Literal["macos", "bubblewrap", "landlock"]
    """本次实际使用的隔离后端。"""

    returncode: int
    """主进程退出码；负数表示被信号终止。"""

    stdout: str
    """标准输出的 UTF-8 预览，非法字节替换为占位字符。"""

    stderr: str
    """标准错误的 UTF-8 预览，非法字节替换为占位字符。"""

    timed_out: bool
    """命令或其输出管道是否超过墙钟预算。"""

    output_truncated: bool
    """任一输出流是否超过保留预算。"""

    duration_s: float
    """从启动到清理结束的实际耗时，单位秒。"""

    @computed_field
    @property
    def ok(self) -> bool:
        """仅在没有超时且退出码为零时返回成功。"""
        return self.returncode == 0 and not self.timed_out
