"""向控制台输出会话启动时的提示词和注册清单。"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable
from typing import TYPE_CHECKING

from .python_tools import PythonTools
from .tools import StatefulTool

if TYPE_CHECKING:
    from .agent.session import Session


def print_session_startup(session: Session, tool_instances: Iterable[StatefulTool]) -> None:
    """打印本次启动后的会话配置，不读取数据或实例化 Python 业务工具。

    Args:
        session: 已完成全部资源和工具注册的会话。
        tool_instances: 当前会话已经绑定的有状态工具实例，不包含普通资源。
    """
    datasets = [dataset.model_dump() for dataset in session.state.datasets.values()]
    tools = session.schemas()
    python_classes = [
        name
        for instance in tool_instances
        if isinstance(instance, PythonTools)
        for name in instance.class_names()
    ]
    sections = [
        f"[Session started] {session.state.session_id}",
        f"=== System prompt ===\n{session.loop.build_system_prompt()}",
        f"=== Datasets ({len(datasets)}) ===\n"
        + (json.dumps(datasets, ensure_ascii=False, indent=2) if datasets else "(none)"),
        f"=== Tools ({len(tools)}) ===\n"
        + "\n".join(
            f"- {schema['function']['name']}: {schema['function']['description']}"
            for schema in tools
        ),
        f"=== Python tools ({len(python_classes)}) ===\n"
        + ("\n".join(f"- {name}" for name in python_classes) if python_classes else "(none)"),
    ]
    print("\n\n".join(sections), file=sys.stderr, flush=True)
