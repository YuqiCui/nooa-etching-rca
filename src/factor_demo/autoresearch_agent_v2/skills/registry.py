"""从宿主批准目录发现技能，并冻结安全读取的文本资源。"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Iterator, Sequence
from pathlib import Path
from weakref import finalize

import yaml

from ..utils.file import WorkspaceFiles
from .typing import SkillDescription, SkillSnapshot

MAX_SKILL_BYTES = 2 * 1024 * 1024
MAX_SKILL_FILES = 128


def _close_roots(roots: dict[Path, WorkspaceFiles]) -> None:
    """释放批准目录描述符，不保留注册目录对象的反向引用。"""
    for files in roots.values():
        files.close()


def snapshot_digest(description: SkillDescription, content: str, resources: dict[str, str]) -> str:
    """对元数据、正文和引用内容计算稳定摘要。"""
    encoded = json.dumps(
        {"description": description.model_dump(), "content": content, "resources": resources},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _description(skill_id: str, source: str) -> SkillDescription:
    """解析 YAML 元信息，不执行正文或资源脚本。"""
    lines = source.splitlines()
    if not lines or lines[0] != "---":
        raise ValueError(f"技能缺少 YAML 元信息：{skill_id}")
    try:
        end = lines.index("---", 1)
    except ValueError as error:
        raise ValueError(f"技能元信息未结束：{skill_id}") from error
    metadata = yaml.safe_load("\n".join(lines[1:end]))
    if not isinstance(metadata, dict):
        raise TypeError(f"技能元信息必须是对象：{skill_id}")
    return SkillDescription(
        skill_id=skill_id,
        name=metadata.get("name"),
        description=metadata.get("description"),
        version=metadata.get("version", "1"),
        requires_tools=metadata.get("requires_tools", []),
    )


class SkillRegistry:
    """按目录名标识技能，仅允许读取显式批准根目录中的文件。"""

    def __init__(self, roots: Sequence[Path] = ()) -> None:
        """建立目录索引，重复技能标识直接报错。

        Args:
            roots: 宿主批准的技能父目录，每个子目录包含 SKILL.md。
        """
        self._entries: dict[str, tuple[Path, str, SkillDescription]] = {}
        self._roots: dict[Path, WorkspaceFiles] = {}
        self._finalizer = finalize(self, _close_roots, self._roots)
        try:
            for root in roots:
                self._discover(root)
        except BaseException:
            self.close()
            raise

    def _discover(self, root: Path) -> None:
        """只发现一层技能目录，拒绝用符号链接扩大批准范围。"""
        approved = root.expanduser().resolve(strict=True)
        if not approved.is_dir():
            raise ValueError("技能根路径必须是目录")
        if approved not in self._roots:
            self._roots[approved] = WorkspaceFiles(approved)
        for child in sorted(approved.iterdir()):
            if child.is_symlink():
                raise ValueError("技能目录不接受符号链接")
            if not child.is_dir() or not (child / "SKILL.md").exists():
                continue
            skill_id = child.name
            if skill_id in self._entries:
                raise ValueError(f"技能标识重复：{skill_id}")
            source = self._read(approved, f"{skill_id}/SKILL.md")
            self._entries[skill_id] = (approved, skill_id, _description(skill_id, source))

    def _read(self, root: Path, path: str) -> str:
        """复用文件工具的目录描述符边界读取 UTF-8 文本。"""
        files = self._roots[root]
        with files.open_parent(path) as (directory, name, _):
            result = files.read_bytes(directory, name)
        if result is None:
            raise FileNotFoundError(path)
        if len(result[0]) > MAX_SKILL_BYTES:
            raise ValueError("技能文本超过大小限制")
        return result[0].decode("utf-8-sig")

    @staticmethod
    def _resource_names(directory: int, prefix: str = "") -> Iterator[str]:
        """通过目录描述符遍历资源，路径被替换时不跟随外部链接。"""
        with os.scandir(directory) as entries:
            names = sorted(entry.name for entry in entries)
        for name in names:
            info = os.stat(name, dir_fd=directory, follow_symlinks=False)
            if stat.S_ISLNK(info.st_mode):
                raise ValueError("技能引用不接受符号链接")
            relative = f"{prefix}{name}"
            if stat.S_ISDIR(info.st_mode):
                child = os.open(
                    name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory
                )
                try:
                    yield from SkillRegistry._resource_names(child, f"{relative}/")
                finally:
                    os.close(child)
            elif stat.S_ISREG(info.st_mode):
                yield relative
            else:
                raise ValueError("技能引用必须是普通文本文件")

    def catalog(self) -> list[SkillDescription]:
        """返回独立元数据，不暴露正文和宿主目录。"""
        return [entry[2].model_copy(deep=True) for entry in self._entries.values()]

    def contains(self, skill_id: str) -> bool:
        """检查标识是否仍在宿主批准的目录中。"""
        return skill_id in self._entries

    def load(self, skill_id: str) -> SkillSnapshot:
        """冻结技能正文及引用资源，拒绝符号链接和超量资源。

        Args:
            skill_id: 目录中已登记的稳定标识。

        Returns:
            含完整内容与哈希的可序列化快照。
        """
        if skill_id not in self._entries:
            raise ValueError(f"未知技能：{skill_id}")
        root, folder, _ = self._entries[skill_id]
        resources: dict[str, str] = {}
        total_bytes = 0
        with self._roots[root].open_parent(f"{folder}/SKILL.md") as (directory, _, _):
            for relative in self._resource_names(directory):
                content = self._read(root, f"{folder}/{relative}")
                total_bytes += len(content.encode())
                if total_bytes > MAX_SKILL_BYTES or len(resources) >= MAX_SKILL_FILES:
                    raise ValueError("技能引用超过资源限制")
                resources[relative] = content
        content = resources.pop("SKILL.md")
        description = _description(skill_id, content)
        return SkillSnapshot(
            description=description,
            content=content,
            resources=resources,
            sha256=snapshot_digest(description, content, resources),
        )

    def close(self) -> None:
        """关闭批准根目录句柄，已冻结的会话正文仍由各会话保存。"""
        self._finalizer()
