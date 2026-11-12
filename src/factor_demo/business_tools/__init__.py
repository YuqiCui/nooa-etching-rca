"""按应用需求显式登记业务工具，内核不会自动导入本包。"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from factor_demo.autoresearch_agent_v2.tools.registry import ToolRegistry


def register_business_tools(registry: "ToolRegistry") -> None:
    """登记已有统计工具，不改变其算法或创建工具实例。"""
    from .statistics import Statistics

    registry.register_python("Statistics", Statistics, methods=Statistics.tool_methods)


__all__ = ["register_business_tools"]
