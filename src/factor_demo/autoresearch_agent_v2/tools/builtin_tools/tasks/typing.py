"""研究工作项工具的输入约束。"""

import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints, model_validator

NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
"""去除首尾空白后仍有内容的文本。"""
Status = Literal["pending", "in_progress", "completed", "deleted"]
"""正常执行状态，以及更新时使用的删除操作。"""


class TaskCreateInput(BaseModel):
    """创建工作项，状态与标识由宿主生成。"""

    model_config = ConfigDict(extra="forbid", strict=True)
    """拒绝未知字段和隐式类型转换。"""

    subject: NonBlank
    """工作项的简短标题。"""

    description: str = ""
    """执行范围和预期产出。"""

    active_form: NonBlank | None = None
    """执行中的显示文本；None 表示未设置。"""

    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    """研究附加信息，顶层 None 不作为持久化值。"""

    @model_validator(mode="after")
    def check_metadata(self):
        """拒绝非有限数字和创建时无意义的顶层空值。"""
        json.dumps(self.metadata, allow_nan=False)
        if any(value is None for value in self.metadata.values()):
            raise ValueError("创建时 metadata 顶层值不能为 null")
        return self


class TaskGetInput(BaseModel):
    """限定在当前研究任务内的工作项查询。"""

    model_config = ConfigDict(extra="forbid", strict=True)
    """拒绝额外范围参数和隐式类型转换。"""

    task_id: NonBlank
    """宿主生成的工作项标识。"""


class TaskUpdateInput(TaskGetInput):
    """仅修改明确提交的字段，保留缺省与显式空值的区别。"""

    subject: NonBlank | None = None
    """新标题，提交时不能为 None。"""

    description: str | None = None
    """新描述，空字符串表示清空，提交时不能为 None。"""

    status: Status | None = None
    """新状态，提交时不能为 None。"""

    active_form: NonBlank | None = None
    """新执行文本，显式 None 表示清空。"""

    metadata: dict[str, JsonValue] | None = None
    """顶层合并补丁，键值为 None 表示移除该键。"""

    @model_validator(mode="after")
    def check_patch(self):
        """拒绝非法空值、非有限数字和混合删除。"""
        fields = self.model_fields_set
        if (
            ("subject" in fields and self.subject is None)
            or ("description" in fields and self.description is None)
            or ("status" in fields and self.status is None)
            or ("metadata" in fields and self.metadata is None)
        ):
            raise ValueError("只有 active_form 接受显式 null")
        json.dumps(self.metadata, allow_nan=False)
        if self.status == "deleted" and fields != {"task_id", "status"}:
            raise ValueError("删除不能同时修改其他字段")
        return self


class TaskListInput(BaseModel):
    """列表不接受范围、排序或过滤参数。"""

    model_config = ConfigDict(extra="forbid", strict=True)
    """拒绝模型传入任务范围。"""
