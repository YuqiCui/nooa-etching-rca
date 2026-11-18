"""在现有工作目录授权和读取记录上执行补丁，不提供额外文件权限。"""

from __future__ import annotations

import errno
import os
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ....utils.file import MAX_FILE_BYTES, FileToolError, _version
from .patch import PatchOperation, apply_hunks, parse_patch
from .tools import _decode, _encode, _file_error_result
from .typing import FileReadState

if TYPE_CHECKING:
    from .tools import FileTools

_MAX_PATCH_BYTES = 1024 * 1024
_MAX_OPERATIONS = 128
_MAX_PLAN_BYTES = 64 * 1024 * 1024
_EXPECTED_ERRORS = (FileToolError, UnicodeError, OSError)
type Snapshot = tuple[bytes, os.stat_result] | None


@dataclass
class PlannedChange:
    """保存只读预检查得到的内容和执行所需权限位。"""

    operation: PatchOperation
    content: bytes | None
    mode: int | None
    complete: bool


def _record(operation: PatchOperation) -> dict[str, Any]:
    """生成不含补丁正文的有界操作记录。"""
    result: dict[str, Any] = {"kind": operation.kind, "path": operation.path}
    if operation.destination is not None:
        result.update(kind="move", destination=operation.destination)
    return result


def _same_snapshot(left: Snapshot, right: Snapshot) -> bool:
    """同时比较内容与版本，判断失败操作是否仍保持原状。"""
    if left is None or right is None:
        return left is right
    return left[0] == right[0] and _version(left[1]) == _version(right[1])


class PatchExecutor:
    """先模拟整份补丁，再在共享路径锁内按序发布；失败不回滚。"""

    def __init__(self, tools: FileTools) -> None:
        """绑定当前会话文件能力，不建立新的读取授权。"""
        self.tools = tools
        self.files = tools.files
        self.snapshots: dict[str, Snapshot] = {}
        self.shadow: dict[str, bytes | None] = {}
        self.complete: dict[str, bool] = {}
        self.modes: dict[str, int | None] = {}
        self.touched: set[str] = set()
        self.plan_bytes = 0
        self.completed: list[dict[str, Any]] = []
        self.affected: list[str] = []

    def execute(self, patch: str) -> dict[str, Any]:
        """限制输入规模并在预检查通过后提交，不向模型回显正文。"""
        if len(patch.encode("utf-8")) > _MAX_PATCH_BYTES:
            raise FileToolError("patch_too_large", "Keep the patch within 1 MiB.")
        try:
            operations = parse_patch(patch)
        except FileToolError as exc:
            return {**exc.result, "phase": "syntax", "completed": [], "affected_paths": []}
        if len(operations) > _MAX_OPERATIONS:
            raise FileToolError("patch_too_large", "Use at most 128 operations per patch.")
        try:
            paths = self._normalize_paths(operations)
        except _EXPECTED_ERRORS as exc:
            return {
                **_file_error_result(exc),
                "phase": "preflight",
                "completed": [],
                "affected_paths": [],
                "skipped": len(operations),
            }
        with self.files.lock_paths(paths):
            changes = []
            for operation in operations:
                try:
                    changes.append(self._plan(operation))
                except _EXPECTED_ERRORS as exc:
                    return {
                        **_file_error_result(exc),
                        "phase": "preflight",
                        "failed_operation": _record(operation),
                        "completed": [],
                        "affected_paths": [],
                        "skipped": len(operations),
                    }
            return self._commit(changes)

    def _normalize_paths(self, operations: list[PatchOperation]) -> list[str]:
        """提前拒绝越界、绝对路径和相互作为父目录的文件目标。"""
        paths = []
        for operation in operations:
            for attribute in ("path", "destination"):
                path = getattr(operation, attribute)
                if path is None:
                    continue
                if Path(path).is_absolute():
                    raise FileToolError("permission_denied", "Use workspace-relative paths.")
                key = str(self.files.relative_path(path))
                setattr(operation, attribute, key)
                paths.append(key)
            if operation.path == operation.destination:
                raise FileToolError("same_target", "Move to a different path.")
        targets = set(paths)
        if any(str(parent) in targets for path in paths for parent in Path(path).parents):
            raise FileToolError("conflicting_paths", "A file target cannot be a parent directory.")
        return paths

    def _budget(self, content: bytes) -> None:
        """限制预检查保留的文件快照和中间结果总量。"""
        if len(content) > MAX_FILE_BYTES:
            raise FileToolError("file_too_large", "Keep text files within the 16 MiB limit.")
        self.plan_bytes += len(content)
        if self.plan_bytes > _MAX_PLAN_BYTES:
            raise FileToolError("patch_too_large", "Split the patch into smaller batches.")

    def _load(self, path: str) -> bytes | None:
        """读取预检查快照，不把内部读取记为模型已读。"""
        if path not in self.shadow:
            snapshot = self.files.snapshot(path)
            self.snapshots[path] = snapshot
            content = snapshot[0] if snapshot is not None else None
            if content is not None:
                self._budget(content)
                _decode(content)
            self.shadow[path] = content
            state = self.tools._reads.get(path)
            self.complete[path] = bool(state and state.complete)
            self.modes[path] = snapshot[1].st_mode if snapshot else None
        return self.shadow[path]

    def _source(self, operation: PatchOperation) -> bytes:
        """检查源文件存在、已读取、版本新鲜和可写权限。"""
        path = operation.path
        content = self._load(path)
        if content is None:
            raise FileToolError("not_found", "Read an existing source file.")
        if operation.kind == "delete" and not self.complete[path]:
            raise FileToolError("read_required", "Read the entire source before deleting.")
        if path not in self.touched:
            state = self.tools._reads.get(path)
            if state is None:
                raise FileToolError("read_required", "Read the source before modifying it.")
            if state.digest != sha256(content).digest():
                raise FileToolError("file_changed", "Read the current source before retrying.")
        mode = self.modes[path]
        if mode is not None and not mode & 0o222:
            raise PermissionError(errno.EACCES, "Source file is read-only")
        return content

    def _plan(self, operation: PatchOperation) -> PlannedChange:
        """在模拟状态中执行一步，整个阶段不修改文件或读取记录。"""
        path = operation.path
        if operation.kind == "add":
            if self._load(path) is not None:
                raise FileToolError("already_exists", "Add File requires an absent target.")
            content = _encode(operation.content)
            complete, mode = True, None
        else:
            before = self._source(operation)
            complete, mode = self.complete[path], self.modes[path]
            if operation.kind == "delete":
                content = None
            else:
                text = _decode(before)
                updated = apply_hunks(text.replace("\r\n", "\n"), operation.hunks)
                # 统一 CRLF 时沿用原格式；混合换行按上游补丁语义统一为 LF。
                rest = text.replace("\r\n", "")
                if "\r\n" in text and "\n" not in rest and "\r" not in rest:
                    updated = updated.replace("\n", "\r\n")
                content = _encode(updated, before)
        if operation.destination is not None and self._load(operation.destination) is not None:
            raise FileToolError("already_exists", "Move requires an absent destination.")
        if content is not None:
            self._budget(content)
        destination = operation.destination or path
        self.shadow[destination] = content
        self.complete[destination] = complete
        self.modes[destination] = mode
        self.touched.update((path, destination))
        if operation.destination is not None:
            self.shadow[path] = None
        return PlannedChange(operation, content, mode, complete)

    def _publish(self, path: str, change: PlannedChange) -> None:
        """按预期快照原子发布一个文件，并同步当前会话的读取状态。"""
        assert change.content is not None
        with self.files.open_parent(path, create=True) as (directory_fd, name, key):
            self.files.replace_bytes(
                directory_fd, name, change.content, self.snapshots[path], mode=change.mode
            )
            self.affected.append(path)
            snapshot = self.files.read_bytes(directory_fd, name)
            if snapshot is None or snapshot[0] != change.content:
                raise FileToolError("file_changed", "Read the target again before retrying.")
            self.snapshots[path] = snapshot
            self.tools._reads[key] = FileReadState(
                digest=sha256(change.content).digest(), complete=change.complete
            )

    def _delete(self, path: str) -> None:
        """通过目录描述符删除预期版本，不跟随符号链接。"""
        snapshot = self.snapshots[path]
        assert snapshot is not None
        with self.files.open_parent(path) as (directory_fd, name, key):
            self.files.delete_bytes(directory_fd, name, snapshot)
            self.affected.append(path)
            self.snapshots[path] = None
            self.tools._reads.pop(key, None)

    def _commit(self, changes: list[PlannedChange]) -> dict[str, Any]:
        """逐步提交，失败时返回已完成操作和无法确定的影响范围。"""
        for index, change in enumerate(changes):
            operation = change.operation
            paths = [operation.path]
            if operation.destination is not None:
                paths.append(operation.destination)
            before = {path: self.snapshots[path] for path in paths}
            try:
                # 移动先检查源版本，再发布目标，最后删除源；中途失败如实报告两边状态。
                if operation.destination is not None:
                    if not _same_snapshot(
                        before[operation.path], self.files.snapshot(operation.path)
                    ):
                        raise FileToolError("file_changed", "Read the source again.")
                    self._publish(operation.destination, change)
                    self._delete(operation.path)
                elif change.content is None:
                    self._delete(operation.path)
                else:
                    self._publish(operation.path, change)
            except _EXPECTED_ERRORS as exc:
                uncertain = []
                for path in paths:
                    try:
                        if not _same_snapshot(self.snapshots[path], self.files.snapshot(path)):
                            uncertain.append(path)
                    except _EXPECTED_ERRORS:
                        uncertain.append(path)
                    self.tools._reads.pop(path, None)
                return {
                    **_file_error_result(exc),
                    "phase": "commit",
                    "failed_operation": _record(operation),
                    "completed": self.completed,
                    "affected_paths": list(dict.fromkeys(self.affected)),
                    "uncertain_paths": uncertain,
                    "skipped": len(changes) - index - 1,
                }
            self.completed.append(_record(operation))
        return {
            "status": "applied",
            "completed": self.completed,
            "affected_paths": list(dict.fromkeys(self.affected)),
        }
