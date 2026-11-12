"""为每个会话维护独立技能集合及有界上下文投影。"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import PurePosixPath
from typing import Any

from .registry import SkillRegistry, snapshot_digest
from .typing import SkillSnapshot


class SkillContext:
    """保存激活快照，不向消息历史重复追加技能正文。"""

    def __init__(
        self,
        registry: SkillRegistry,
        *,
        available_tools: Iterable[str] = (),
        snapshots: Sequence[dict[str, Any]] = (),
    ) -> None:
        """恢复可信快照并核对当前能力上限。

        Args:
            registry: 宿主批准的技能目录。
            available_tools: 当前会话实际获准使用的工具标识。
            snapshots: 会话检查点中的既有快照。
        """
        self.registry = registry
        self._tools = frozenset(available_tools)
        self._active: dict[str, SkillSnapshot] = {}
        for saved in snapshots:
            snapshot = SkillSnapshot.model_validate(saved)
            self._validate(snapshot)
            skill_id = snapshot.description.skill_id
            if skill_id in self._active:
                raise ValueError(f"技能快照标识重复：{skill_id}")
            self._active[skill_id] = snapshot

    def _validate(self, snapshot: SkillSnapshot) -> None:
        """核对授权、内容完整性和引用名称，防止伪造恢复状态。"""
        description = snapshot.description
        if not self.registry.contains(description.skill_id):
            raise ValueError(f"技能已不在批准目录：{description.skill_id}")
        expected = snapshot_digest(description, snapshot.content, snapshot.resources)
        if expected != snapshot.sha256:
            raise ValueError("技能快照哈希不匹配")
        for name in snapshot.resources:
            self._resource_path(name)
        missing = set(description.requires_tools) - self._tools
        if missing:
            raise ValueError(f"技能缺少工具：{', '.join(sorted(missing))}")

    @staticmethod
    def _resource_path(path: str) -> str:
        """只接受技能内规范的相对 POSIX 路径。"""
        relative = PurePosixPath(path)
        if (
            not path
            or "\x00" in path
            or "\\" in path
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != path
        ):
            raise ValueError("技能资源路径必须位于技能目录内")
        return path

    def load(self, skill_id: str) -> SkillSnapshot:
        """幂等激活一个技能，既有快照不被部署目录更新覆盖。"""
        if skill_id not in self._active:
            snapshot = self.registry.load(skill_id)
            self._validate(snapshot)
            self._active[skill_id] = snapshot
        return self._active[skill_id].model_copy(deep=True)

    def preload(self, skill_ids: Sequence[str]) -> None:
        """在首次模型请求前加载父 agent 明确选择的技能。"""
        previous = self._active.copy()
        try:
            for skill_id in skill_ids:
                self.load(skill_id)
        except Exception:
            self._active = previous
            raise

    def snapshots(self) -> list[dict[str, Any]]:
        """导出独立快照，供会话检查点与子任务选择使用。"""
        return [snapshot.model_dump(mode="json") for snapshot in self._active.values()]

    def catalog(self) -> list[dict[str, Any]]:
        """返回技能目录及本会话激活标记，不返回引用正文。"""
        return [
            {**description.model_dump(), "loaded": description.skill_id in self._active}
            for description in self.registry.catalog()
        ]

    def read_resource(self, skill_id: str, path: str) -> str:
        """只读取已激活快照中的引用文件，不再次访问宿主路径。"""
        if skill_id not in self._active:
            raise ValueError("请先加载技能")
        path = self._resource_path(path)
        snapshot = self._active[skill_id]
        if path == "SKILL.md":
            return snapshot.content
        try:
            return snapshot.resources[path]
        except KeyError as error:
            raise ValueError("技能资源不存在") from error

    def instructions(self) -> str:
        """按当前激活集合构建正文，不包含长引用文件内容。"""
        catalog = json.dumps(self.catalog(), ensure_ascii=False)
        parts = ["# 可用技能\n以下是宿主批准的技能目录。\n" + catalog]
        for skill_id, snapshot in self._active.items():
            parts.append(
                f"# 已加载技能：{skill_id}\n版本：{snapshot.description.version}；"
                f"sha256：{snapshot.sha256}\n{snapshot.content}"
            )
        return "\n\n".join(parts)
