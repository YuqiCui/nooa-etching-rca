"""Notebook 工具共用的参数约束与执行结果。"""

from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field


def _validate_notebook_name(value: str) -> str:
    """拒绝路径、控制字符和容易混淆的文件扩展名。"""
    if (
        value != value.strip()
        or value in {".", ".."}
        or value.endswith(".ipynb")
        or any(char in "/\\" or ord(char) < 32 for char in value)
        or len(value.encode("utf-8")) > 240
    ):
        raise ValueError("Use a trimmed notebook name without paths or the .ipynb extension")
    return value


NotebookName = Annotated[
    str, Field(min_length=1, max_length=80), AfterValidator(_validate_notebook_name)
]
"""当前会话内唯一的业务名称，允许中文，不接受文件路径。"""

CellId = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")]
"""nbformat 定义的稳定 cell 标识，不随插入顺序变化。"""

Revision = Annotated[int, Field(ge=1)]
"""读取或编辑工具返回的明确源码版本。"""

MAX_CELL_SOURCE_CHARS = 1_000_000
"""单个 cell 源码和补丁片段的字符上限。"""

CellSource = Annotated[str, Field(max_length=MAX_CELL_SOURCE_CHARS)]
"""单个 cell 的完整源码，允许空字符串。"""


class CellWriteRequest(BaseModel):
    """批量插入中的单个 Python cell。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    source: CellSource
    """完整源码；写入时不执行。"""


class CellRunRequest(BaseModel):
    """批量执行中的 cell 标识与明确源码版本。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    cell_id: CellId
    """写入或读取工具返回的稳定标识。"""

    revision: Revision
    """要执行的当前源码版本。"""


CellWrites = Annotated[list[CellWriteRequest], Field(min_length=1, max_length=100)]
"""一次插入 1 至 100 个 cell，保持传入顺序。"""

CellRuns = Annotated[list[CellRunRequest], Field(min_length=1, max_length=100)]
"""一次执行 1 至 100 项，保持传入顺序。"""


class CellExecution(BaseModel):
    """保留一次真实 kernel 请求的结果与输出预算状态。"""

    status: Literal["completed", "error", "timeout", "crashed"]
    """执行完成、Python 异常、超时或 kernel 退出。"""

    execution_count: int | None = None
    """kernel 返回的执行序号；尚未收到时为 None。"""

    outputs: list[dict[str, Any]] = Field(default_factory=list)
    """符合 nbformat 的有界输出，包括文本、富媒体和异常。"""

    output_truncated: bool = False
    """是否因输出预算省略了部分消息。"""
