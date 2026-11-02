"""向用户输出最终结果的核心工具。"""


def finish(content: str) -> str:
    """打印最终结果，由 Agent loop 结束当前轮次。

    Args:
        content: 要发送给用户的最终内容。

    Returns:
        已输出的最终内容。
    """
    print(content, flush=True)
    return content
