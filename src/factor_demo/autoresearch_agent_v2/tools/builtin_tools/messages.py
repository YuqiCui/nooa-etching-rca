"""向运行时提交当前工作的结束内容。"""


def finish(content: str) -> str:
    """返回结束内容，由运行时决定投递给父会话还是用户。

    Args:
        content: 当前工作的结果正文，格式由任务或 skill 决定。

    Returns:
        尚待运行时路由的完整内容。
    """
    return content
