"""按显式清单导入内置 Python 工具，不扫描目录。"""

from importlib import import_module

_BUILTIN_MODULES = (".builtin_tools.statistics",)
"""仅登记类定义，不在导入阶段创建实例或加载方法 Schema。"""


def ensure_registered() -> None:
    """导入内置定义，模块缓存保证正常导入不会重复登记。"""
    for module_name in _BUILTIN_MODULES:
        import_module(module_name, package=__package__)
