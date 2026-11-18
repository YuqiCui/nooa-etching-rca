"""通用委派及消息传递所需的数据类型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ..agent import Session, SessionState

ExecutionStatus = Literal["queued", "running", "completed", "failed", "cancelled", "interrupted"]
ACTIVE_STATUSES = ("queued", "running", "interrupted")


@dataclass(frozen=True)
class ChildRequest:
    """由运行时生成身份和权限的子会话启动请求。"""

    state: SessionState
    task: str
    skills: tuple[str, ...]
    allow_tools: tuple[str, ...]
    parent: Session
    assignment_id: str
    description: str = ""
    """父 Agent 调用该子会话的目的，与实际任务正文分开保存。"""


@dataclass(frozen=True)
class RuntimeLimits:
    """限定并行规模、嵌套深度和消息大小。"""

    max_concurrent: int = 4
    max_depth: int = 2
    max_task_chars: int = 100_000
    result_preview_chars: int = 2_000
    inbox_batch_size: int = 8
    max_workers: int = 64
    wait_timeout_s: float = 4 * 60 * 60
    max_finish_repairs: int = 3

    def __post_init__(self) -> None:
        """拒绝使运行时无法推进的预算。"""
        if (
            min(
                self.max_concurrent,
                self.max_depth,
                self.max_task_chars,
                self.result_preview_chars,
                self.inbox_batch_size,
                self.max_workers,
                self.max_finish_repairs,
            )
            < 1 or self.wait_timeout_s <= 0
        ):
            raise ValueError("运行时计数限制必须为正整数，等待期限必须大于零")
