"""以有界原文、精确替换和简短确认减少文件编辑的模型上下文。"""

from __future__ import annotations

import errno
import io
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from functools import wraps
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, ClassVar

from pydantic import Field

from ....utils.file import FileToolError, WorkspaceFiles
from ...base import ToolFactory
from .typing import FileReadState

if TYPE_CHECKING:
    from ....agent.session import Session

MAX_READ_BYTES = 12 * 1024
"""包含行号和分页信息的最终读取结果预算。"""

_MAX_MATCH_LINES = 8


def _recover_file_errors(function: Callable[..., Any]) -> Callable[..., Any]:
    """将预期文件错误转为有界结果，保留关闭错误和意外故障的传播。"""

    @wraps(function)
    def invoke(self: FileTools, *args: Any, **kwargs: Any) -> Any:
        """先检查生命周期，再执行受授权约束的文件操作。"""
        self.files.require_open()
        try:
            return function(self, *args, **kwargs)
        except FileToolError as exc:
            return exc.result
        except UnicodeError:
            return {"error": "invalid_text", "hint": "Use valid UTF-8 text without binary data."}
        except OSError as exc:
            code, hint = {
                errno.ENOENT: ("not_found", "Check the file path."),
                errno.EISDIR: ("not_file", "Use a regular text file."),
                errno.EACCES: ("permission_denied", "Use a file with the required permissions."),
                errno.EPERM: ("permission_denied", "Use a file with the required permissions."),
                errno.EEXIST: ("file_changed", "Read the target again before retrying."),
            }.get(exc.errno, ("io_error", "Check file permissions and available storage."))
            return {"error": code, "hint": hint}

    return invoke


def _decode(content: bytes) -> str:
    """严格解码 UTF-8，隐藏格式用 BOM，并拒绝含 NUL 的二进制内容。"""
    text = content.decode("utf-8-sig")
    if "\x00" in text:
        raise FileToolError("invalid_text", "Use UTF-8 text without NUL bytes.")
    return text


def _encode(text: str, previous: bytes = b"") -> bytes:
    """原样编码内容，覆盖和编辑已有文件时保留其 UTF-8 BOM。"""
    if "\x00" in text:
        raise FileToolError("invalid_text", "Use UTF-8 text without NUL bytes.")
    content = text.encode("utf-8")
    if previous.startswith(b"\xef\xbb\xbf") and not content.startswith(b"\xef\xbb\xbf"):
        content = b"\xef\xbb\xbf" + content
    return content


def _match_details(text: str, needle: str) -> tuple[int, list[int]]:
    """统计含重叠位置的全部匹配，最多保留八个起始行。"""
    lines = []
    start = 0
    count = 0
    while True:
        index = text.find(needle, start)
        if index < 0:
            break
        count += 1
        if len(lines) < _MAX_MATCH_LINES:
            prefix = text[:index].replace("\r\n", "\n").replace("\r", "\n")
            lines.append(prefix.count("\n") + 1)
        start = index + 1
    return count, lines


class FileTools:
    """同一会话共享读取记录，并在关闭时撤销文件访问。"""

    tool_names: ClassVar[tuple[str, ...]] = ("read_file", "write_file", "edit_file")
    """启动前可检查的公开工具名称。"""

    namespace: ClassVar[str] = "files"
    """会话工具组标识；模型工具保留 read_file 等无前缀名称。"""

    def __init__(self, workdir: Path) -> None:
        """绑定唯一的可读写工作目录并创建独立读取记录。

        Args:
            workdir: 已存在、由宿主授权的会话目录。

        Raises:
            OSError: 工作目录无法打开。
        """
        self.files = WorkspaceFiles(workdir)
        self._reads: dict[str, FileReadState] = {}

    @classmethod
    @asynccontextmanager
    async def bind(cls, session: Session) -> AsyncIterator[FileTools]:
        """创建当前会话的工具组，退出时关闭目录描述符并清除读取记录。

        Args:
            session: 提供已授权 workdir 的会话。

        Yields:
            三个文件方法共享读取记录的工具实例。
        """
        tools = cls(session.state.workdir)
        try:
            yield tools
        finally:
            tools.files.close()
            tools._reads.clear()

    def register_tools(self, factory: ToolFactory) -> None:
        """注册最小接口，并让参数错误作为结果交给模型修正。

        Args:
            factory: 当前会话启动期间的独立工具注册表。
        """
        for name, function, description in (
            (
                "read_file",
                self.read_file,
                "读取文本文件，可指定起始行和行数。返回带行号的原文；编辑时不要复制行号前缀。",
            ),
            (
                "write_file",
                self.write_file,
                "创建或完整覆盖文本文件。覆盖前须完整读取已有文件；局部修改优先使用 edit_file。",
            ),
            (
                "edit_file",
                self.edit_file,
                (
                    "先读取目标文件，再精确替换文本。old_string 须唯一匹配；"
                    "replace_all=true 替换全部。new_string 为空表示删除。"
                ),
            ),
        ):
            factory.register_callable(
                name, function, description=description, recover_validation_errors=True
            )

    def _remember_range(self, key: str, digest: bytes, start: int, end: int, total: int) -> None:
        """只累计相同内容版本中实际交付的完整行，连续分页可建立完整读取记录。"""
        state = self._reads.get(key)
        if state is None or state.digest != digest:
            state = FileReadState(digest=digest)
            self._reads[key] = state
        merged: list[tuple[int, int]] = []
        for left, right in sorted([*state.ranges, (start, end)]):
            if merged and left <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(right, merged[-1][1]))
            else:
                merged.append((left, right))
        state.ranges = merged
        state.complete = state.complete or total == 0 or merged == [(1, total)]

    @_recover_file_errors
    def read_file(
        self,
        path: str,
        offset: Annotated[int, Field(ge=1)] = 1,
        limit: Annotated[int, Field(ge=1)] = 200,
    ) -> str | dict[str, Any]:
        """返回完整行和可继续读取的位置，最多输出 12 KiB。

        Args:
            path: 工作目录内的相对路径或绝对路径。
            offset: 从 1 开始的起始行。
            limit: 最多返回的行数，默认 200。

        Returns:
            带行号的原文和分页信息，或有界的可恢复错误。
        """
        if offset < 1 or limit < 1:
            raise FileToolError("invalid_argument", "offset and limit must be positive integers.")
        with self.files.open_parent(path) as (directory_fd, name, key):
            snapshot = self.files.read_bytes(directory_fd, name)
            if snapshot is None:
                raise FileToolError("not_found", "Check the file path.")
            content, _ = snapshot
            text = _decode(content)
            lines = io.StringIO(text, newline="").readlines()
            total = len(lines)
            digest = sha256(content).digest()
            if total == 0 and offset == 1:
                self._remember_range(key, digest, 1, 0, 0)
                return "empty file"
            if offset > total:
                raise FileToolError(
                    "offset_out_of_range", "Choose an offset within the file.", total_lines=total
                )
            output = []
            # 为行范围、总行数和 next_offset 留出空间，最终结果仍受完整预算约束。
            remaining = MAX_READ_BYTES - 128
            for index in range(offset - 1, min(total, offset - 1 + limit)):
                line = f"{index + 1}\t{lines[index]}"
                size = len(line.encode("utf-8"))
                if size > remaining:
                    if not output:
                        raise FileToolError(
                            "line_too_long",
                            "Use a streaming tool to inspect this line.",
                            line=index + 1,
                        )
                    break
                output.append(line)
                remaining -= size
            end = offset + len(output) - 1
            result = f"lines {offset}-{end}/{total}\n" + "".join(output)
            if end < total:
                result += f"\nnext_offset={end + 1}"
            self._remember_range(key, digest, offset, end, total)
            return result

    @_recover_file_errors
    def write_file(self, path: str, content: str) -> str | dict[str, Any]:
        """新建或完整覆盖文件，成功后只返回简短确认。

        Args:
            path: 工作目录内的目标文件，可包含尚未创建的父目录。
            content: 完整 UTF-8 文本，不自动增加末尾换行。

        Returns:
            created、written 或 unchanged；前置条件不满足时返回错误。
        """
        encoded = _encode(content)
        with self.files.open_parent(path, create=True) as (directory_fd, name, key):
            snapshot = self.files.read_bytes(directory_fd, name)
            if snapshot is not None:
                state = self._reads.get(key)
                if state is None or not state.complete:
                    raise FileToolError("read_required", "Read the entire file before overwriting.")
                if state.digest != sha256(snapshot[0]).digest():
                    raise FileToolError(
                        "file_changed", "Read the entire current file before retrying."
                    )
                encoded = _encode(content, snapshot[0])
                if encoded == snapshot[0]:
                    return "unchanged"
            self.files.replace_bytes(directory_fd, name, encoded, snapshot)
            self._reads[key] = FileReadState(digest=sha256(encoded).digest(), complete=True)
            return "created" if snapshot is None else "written"

    @_recover_file_errors
    def edit_file(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str | dict[str, Any]:
        """精确替换当前内容中的匹配片段，不回传修改内容或 diff。

        Args:
            path: 工作目录内已经读取过的文件。
            old_string: 非空的原文字面片段，默认必须唯一匹配。
            new_string: 替换文本，空字符串表示删除。
            replace_all: 是否替换所有不重叠匹配，默认 False。

        Returns:
            简短的编辑确认，外部变更标记，或包含候选行号的可恢复错误。
        """
        if not old_string:
            raise FileToolError("invalid_argument", "old_string must not be empty.")
        with self.files.open_parent(path) as (directory_fd, name, key):
            snapshot = self.files.read_bytes(directory_fd, name)
            if snapshot is None:
                raise FileToolError("not_found", "Check the file path.")
            state = self._reads.get(key)
            if state is None:
                raise FileToolError("read_required", "Read the target range before editing.")
            content, _ = snapshot
            text = _decode(content)
            # 统一 CRLF 文件接受模型常用的 LF 片段，写回时保留原有换行格式。
            remaining_newlines = text.replace("\r\n", "")
            if "\r\n" in text and "\n" not in remaining_newlines and "\r" not in remaining_newlines:
                old_string = old_string.replace("\r\n", "\n").replace("\n", "\r\n")
                new_string = new_string.replace("\r\n", "\n").replace("\n", "\r\n")
            matches = text.count(old_string)
            if matches == 0:
                raise FileToolError("no_match", "Read the target range and copy the exact text.")
            if not replace_all and text.find(old_string, text.find(old_string) + 1) >= 0:
                count, lines = _match_details(text, old_string)
                raise FileToolError(
                    "ambiguous_match",
                    "Include surrounding text or set replace_all=true.",
                    matches=count,
                    lines=lines,
                )
            changed = state.digest != sha256(content).digest()
            suffix = "; changed_since_read=true" if changed else ""
            if old_string == new_string:
                return "unchanged" + suffix
            updated = text.replace(old_string, new_string, -1 if replace_all else 1)
            encoded = _encode(updated, content)
            self.files.replace_bytes(directory_fd, name, encoded, snapshot)
            self._reads[key] = FileReadState(
                digest=sha256(encoded).digest(), complete=state.complete and not changed
            )
            message = f"edited {matches} matches" if replace_all else "edited"
            return message + suffix
