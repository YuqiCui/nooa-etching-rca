"""提供固定工作空间内的 Shell 执行和基于 Shell 的 Python 执行。"""

from .config import SandboxConfig
from .sandbox import Sandbox
from .typing import SandboxResult

__all__ = ["Sandbox", "SandboxConfig", "SandboxResult"]
