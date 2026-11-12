"""解析会话工具选择，并在全部调用入口复用同一授权结果。"""

from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

type AllowTools = Literal["all"] | Sequence[str]


class ToolPolicy:
    """将显式选择解析为不可变的工具名称集合。"""

    def __init__(
        self,
        allow_tools: AllowTools = "all",
        available_names: Iterable[str] = (),
        providers: Mapping[str, Iterable[str]] | None = None,
    ) -> None:
        """校验选择，未知工具或 provider 不会静默忽略。

        Args:
            allow_tools: all 或明确的工具/provider 名称列表。
            available_names: 应用已登记且宿主允许的工具名称。
            providers: provider 选择名称到公开工具名的映射。

        Raises:
            ValueError: 选择格式不合法或包含未知名称。
        """
        available = frozenset(available_names)
        self.is_all = allow_tools == "all"
        if self.is_all:
            self.allowed_names = available
            return
        if isinstance(allow_tools, str):
            raise TypeError("allow_tools 只能为 all 或工具名称列表")
        selected: set[str] = set()
        for name in allow_tools:
            if not isinstance(name, str) or not name:
                raise ValueError("工具名称必须是非空字符串")
            if name in available:
                selected.add(name)
            elif providers is not None and name in providers:
                expanded = set(providers[name]) & available
                if not expanded:
                    raise ValueError(f"provider 没有当前可用的工具：{name}")
                selected.update(expanded)
            else:
                raise ValueError(f"未知或未授权的工具：{name}")
        self.allowed_names = frozenset(selected)

    def include_registered(self, name: str) -> None:
        """兼容宿主在 all 模式显式登记旧工具，受限选择保持不变。"""
        if self.is_all:
            self.allowed_names = self.allowed_names | {name}

    def allows(self, name: str) -> bool:
        """判断一个公开工具名称是否获得当前会话授权。"""
        return name in self.allowed_names

    def require(self, name: str) -> None:
        """拒绝通过直接调用或桥接访问未授权工具。"""
        if not self.allows(name):
            raise PermissionError(f"当前会话未获准使用工具：{name}")
