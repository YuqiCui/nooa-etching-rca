"""独立于模型服务和执行资源的会话状态。"""

from copy import deepcopy
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DatasetRegistration(BaseModel):
    """只用于会话提示词的数据集登记信息。"""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True, frozen=True)
    """校验登记字段并保留不可变记录，不接受额外元数据。"""

    name: str = Field(min_length=1, max_length=200, description="当前会话内的数据集名称。")
    """同名登记更新原条目，不新增重复提示。"""

    path: str = Field(min_length=1, max_length=4096, description="数据文件、目录或 URI 的位置。")
    """原样保留的位置说明；相对路径以会话工作目录为基准。"""

    description: str = Field(min_length=1, max_length=1000, description="数据集的简短描述。")
    """用于帮助模型选择数据，不包含数据预览或自动提取的结构。"""


class HistoryCompaction(BaseModel):
    """记录替代历史前缀的摘要及生成时的消息边界。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    """拒绝未知字段，避免单独修改摘要边界。"""

    summary: str = Field(min_length=1)
    """供后续模型继续任务的摘要，不是新的用户指令或原始证据。"""

    end_index: int = Field(ge=1)
    """摘要覆盖的原始消息前缀长度，后续消息仍按原顺序发送。"""

    source_message_count: int = Field(ge=1)
    """生成摘要时的原始消息总数，用于识别前缀变化后失效的旧 thinking。"""


class SessionState(BaseModel):
    """State of one agent conversation within a task."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    """Reject unknown session fields and validate field assignments."""

    session_id: str = Field(min_length=1)
    """Unique identifier of this independent conversation."""

    agent_id: str = Field(min_length=1)
    """Identifier of the agent participating in this conversation."""

    parent_agent_id: str | None = Field(default=None, min_length=1)
    """Identifier of the parent agent; None for an agent without a parent."""

    task_id: str = Field(min_length=1)
    """Identifier of the task containing this session and any sibling sessions."""

    workdir: Path
    """Task workspace shared by its agents; storing this path grants no file access."""

    assignment_id: str | None = None
    """宿主分配的通用委派标识；普通独立会话可以不设置。"""

    skill_snapshots: list[dict[str, Any]] = Field(default_factory=list)
    """当前激活技能的可恢复快照，由技能加载器校验。"""

    consumed_event_ids: list[str] = Field(default_factory=list)
    """已写入消息历史的宿主事件，恢复时用于去重。"""

    messages: list[dict[str, Any]] = Field(default_factory=list)
    """Ordered, complete message payloads, including provider-specific reasoning and tool fields."""

    datasets: dict[str, DatasetRegistration] = Field(default_factory=dict)
    """按名称保存的数据集提示信息，随会话序列化恢复，不授予文件或网络访问权限。"""

    compaction: HistoryCompaction | None = None
    """最近一次成功的压缩检查点；None 表示模型继续使用完整历史。"""

    @model_validator(mode="after")
    def check_dataset_names(self) -> "SessionState":
        """确保序列化字典的键与条目的名称一致。"""
        if any(name != dataset.name for name, dataset in self.datasets.items()):
            raise ValueError("数据集登记键必须与名称一致")
        return self

    @model_validator(mode="after")
    def check_compaction_bounds(self) -> "SessionState":
        """确保压缩检查点只引用当前保存的消息。"""
        if self.compaction is not None and not (
            self.compaction.end_index <= self.compaction.source_message_count <= len(self.messages)
        ):
            raise ValueError("压缩检查点超出消息历史范围")
        return self

    def get_messages(self) -> list[dict[str, Any]]:
        """Return a complete, independent copy of the conversation history.

        Returns:
            All messages in order, retaining every field and nested value without
            truncation, compression, or conversion between provider formats.
        """
        return deepcopy(self.messages)
