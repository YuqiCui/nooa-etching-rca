"""将工具类详情渲染为供模型阅读的 Python 接口文档。"""

from textwrap import indent

from .typing import ClassDetails


def _render_docstring(docstring: str) -> str:
    """保留完整文档及换行，转义 Python 字符串中的反斜杠和三引号。"""
    escaped = docstring.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
    if "\n" in escaped:
        return f'"""{escaped}\n"""'
    return f'"""{escaped}"""'


def render_class_documentation(details: ClassDetails) -> str:
    """按 Python 类、方法签名和完整 docstring 展示已加载的接口。

    Args:
        details: 指定工具类及全部公开方法的完整详情。

    Returns:
        一个 Python 代码块；省略真实实现及内部 JSON Schema。
        方法统一使用同步签名，与执行进程中 ClassName.method 的调用形式一致。
    """
    sections = [
        f"class {details.class_name}:",
        indent(_render_docstring(details.docstring), "    "),
    ]
    for method in details.methods:
        sections.append(
            f"\n    def {method.signature}:\n"
            f"{indent(_render_docstring(method.docstring), '        ')}\n"
            "        ..."
        )
    return "```python\n" + "\n".join(sections) + "\n```"
