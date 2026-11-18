"""保存用户确认的 Router 与通用 Worker 工具开关。"""

from collections.abc import Iterable
from typing import Literal

ROUTER_TOOLS = (
    "skill__list",
    "skill__load",
    "skill__read_resource",
    "StartWorker",
    "AssignTask",
    "CheckWorkerStatus",
    "WaitWorkers",
    "CollectWorkerResult",
    "read_file",
    "write_file",
    "edit_file",
    "apply_patch",
    "register_dataset",
    "report",
    "finish",
)

WORKER_TOOLS = (
    "StartWorker",
    "AssignTask",
    "CheckWorkerStatus",
    "WaitWorkers",
    "CollectWorkerResult",
    "read_file",
    "write_file",
    "edit_file",
    "apply_patch",
    "sandbox__run_shell",
    "notebook__create_notebook",
    "notebook__list_notebooks",
    "notebook__write_new_cell",
    "notebook__write_cells",
    "notebook__list_cells",
    "notebook__read_cell",
    "notebook__edit_cell",
    "notebook__exec_cell",
    "notebook__run_cells",
    "python__describe_class",
    "Statistics.summarize",
    "Statistics.pearson_correlation",
    "register_dataset",
    "report",
    "finish",
)

DISCUSSION_TOOLS = ("finish", "report", "read_file")
"""仅讨论模式继续采用只读子集，不因 Router 可以写文件而开放写入。"""


def role_tool_names(role: Literal["router", "worker"], available: Iterable[str]) -> tuple[str, ...]:
    """选取宿主实际登记且经过审批的能力，不推断或创建工具。

    Args:
        role: 当前宿主需要装配的角色。
        available: 宿主批准并登记的工具名称，包括 Python 业务方法。

    Returns:
        该角色可用的工具名称；新登记的工具不会自动获得权限。

    Raises:
        ValueError: 角色名称不受支持。
    """
    if role not in {"router", "worker"}:
        raise ValueError("工具角色只能为 router 或 worker")
    approved = ROUTER_TOOLS if role == "router" else WORKER_TOOLS
    names = set(available)
    return tuple(name for name in approved if name in names)
