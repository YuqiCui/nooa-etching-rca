"""在工具执行期间传递宿主调用标识，不暴露为模型参数。"""

from contextvars import ContextVar

execution_id: ContextVar[str | None] = ContextVar("tool_execution_id", default=None)
"""当前执行的稳定标识；宿主未提供时不进行执行去重。"""
