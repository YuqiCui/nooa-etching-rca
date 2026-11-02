"""Python 工具的详情及执行进程提交的调用请求。"""

from typing import Any

from pydantic import BaseModel, ConfigDict


class MethodDetails(BaseModel):
    """查看工具类之后才提供的方法完整接口。"""

    name: str
    """在类代理上访问的方法名。"""

    signature: str
    """包含参数、默认值和返回类型的 Python 签名。"""

    docstring: str
    """未经摘要或截断的完整方法文档字符串。"""

    input_schema: dict[str, Any]
    """从同一方法生成并用于实际调用校验的参数 Schema。"""

    output_schema: dict[str, Any]
    """返回值的结构与字段约束。"""


class ClassDetails(BaseModel):
    """一个已加载工具类的全部公开方法详情。"""

    class_name: str
    """可在 Python 代码中直接使用的类名。"""

    docstring: str
    """工具类的功能和方法概览。"""

    methods: list[MethodDetails]
    """显式公开的全部方法，保持工具声明的顺序。"""


class PythonToolCall(BaseModel):
    """从不可信执行进程提交的单次方法调用。"""

    model_config = ConfigDict(extra="forbid", strict=True)
    """拒绝未知路由字段及隐式转换。"""

    class_name: str
    """必须已在当前会话查看详情的工具类名。"""

    method_name: str
    """必须在工具显式公开清单中的方法名。"""

    args: list[Any]
    """按 Python 调用顺序传入的位置参数。"""

    kwargs: dict[str, Any]
    """按名称传入的关键字参数。"""
