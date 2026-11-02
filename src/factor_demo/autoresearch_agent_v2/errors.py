"""区分工具协议错误与工具内部执行异常。"""


class UnknownToolError(KeyError):
    """请求的工具名不在当前会话的注册表中。"""
