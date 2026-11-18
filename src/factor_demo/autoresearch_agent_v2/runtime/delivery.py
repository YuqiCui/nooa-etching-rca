"""验证 Worker 交付清单、状态与实际产物引用。"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Deliverable(BaseModel):
    """当前 Assignment 的一个交付项。"""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    id: str
    status: Literal["provided", "unavailable"]
    content: str = Field(min_length=1)
    artifact_refs: list[str]


class WorkerResult(BaseModel):
    """执行终止结果，不代表证据已经验收。"""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    outcome: Literal["completed", "failed", "blocked", "cancelled"]
    summary: str = Field(min_length=1)
    deliverables: list[Deliverable]
    limitations: list[str]


def validate_delivery(content: str, assignment: dict, workdir: Path) -> dict:
    """验证格式及引用，保持收集响应在通用输出上限内。"""
    if len(content.encode()) > 12_000:
        raise ValueError("交付记录超过 12000 字节；长内容请保存到产物")
    result = WorkerResult.model_validate_json(content)
    expected = {item["id"] for item in assignment["deliver_list"]}
    actual = [item.id for item in result.deliverables]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("每个已分配的交付 ID 必须恰好出现一次")
    if result.outcome == "completed" and any(
        item.status != "provided" for item in result.deliverables
    ):
        raise ValueError("completed 要求全部交付项为 provided")
    root = workdir.resolve(strict=True)
    for item in result.deliverables:
        for reference in item.artifact_refs:
            path = Path(reference)
            path = (root / path).resolve(strict=True) if not path.is_absolute() else path.resolve(
                strict=True
            )
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError("产物引用必须指向真实存在的授权工作目录文件")
    return result.model_dump(mode="json")


def terminal_result(assignment: dict, outcome: str, reason: str) -> dict:
    """宿主异常终止时保留每个交付 ID，不冒称已经交付。"""
    return {
        "outcome": outcome,
        "summary": reason[:1000],
        "deliverables": [
            {"id": item["id"], "status": "unavailable", "content": "执行已终止，原因见 summary。",
             "artifact_refs": []}
            for item in assignment["deliver_list"]
        ],
        "limitations": ["执行已终止；已写入的文件和会话检查点仍保留，交付内容需重新核对。"],
    }
