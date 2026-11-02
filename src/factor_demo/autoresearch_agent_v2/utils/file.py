"""通过目录描述符约束文件访问，并原子发布完整文件。"""

from __future__ import annotations

import errno
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Lock, RLock
from typing import Any
from uuid import uuid4
from weakref import WeakValueDictionary

MAX_FILE_BYTES = 16 * 1024 * 1024
"""单个文本文件的宿主内存处理上限，不是模型输出预算。"""

_FILE_LOCKS = WeakValueDictionary()
_REGISTRY_LOCK = Lock()


class FileToolError(Exception):
    """携带可交给模型修正的文件操作错误。"""

    def __init__(self, code: str, hint: str, **details: Any) -> None:
        """保存不包含文件内容的错误结果。

        Args:
            code: 稳定的错误码。
            hint: 简短的修正提示。
            **details: 候选行号等有界诊断信息。
        """
        super().__init__(code)
        self.result = {"error": code, **details, "hint": hint}


def _file_lock(root: Path, relative: Path) -> Any:
    """让同进程内指向同一文件的工具实例共享锁，空闲锁自动回收。"""
    key = (str(root), str(relative))
    with _REGISTRY_LOCK:
        lock = _FILE_LOCKS.get(key)
        if lock is None:
            lock = RLock()
            _FILE_LOCKS[key] = lock
        return lock


def _version(info: os.stat_result) -> tuple[int, ...]:
    """使用 inode 和高精度元信息识别读取期间的变化。"""
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


class WorkspaceFiles:
    """持有会话根目录描述符，所有路径逐级拒绝符号链接。"""

    def __init__(self, workdir: Path) -> None:
        """打开宿主授予的工作目录，不创建或扩大授权范围。

        Args:
            workdir: 已存在的会话工作目录。

        Raises:
            OSError: 目录不存在或无法打开。
        """
        self.root = workdir.expanduser().resolve(strict=True)
        self._root_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)

    def require_open(self) -> None:
        """拒绝会话关闭后继续使用已经借出的工具实例。

        Raises:
            RuntimeError: 根目录描述符已关闭。
        """
        if self._root_fd is None:
            raise RuntimeError("File tools are closed")

    def close(self) -> None:
        """释放根目录描述符，重复关闭不产生副作用。"""
        if self._root_fd is not None:
            os.close(self._root_fd)
            self._root_fd = None

    @contextmanager
    def open_parent(self, path: str, *, create: bool = False) -> Iterator[tuple[int, str, str]]:
        """锁定文件，并通过可信根目录逐级打开其父目录。

        Args:
            path: 根目录内的相对路径或绝对路径，不接受父目录跳转。
            create: 是否创建缺失的父目录，仅供 write_file 使用。

        Yields:
            父目录描述符、文件名和规范化的相对路径键。

        Raises:
            FileToolError: 路径为空、越界或包含符号链接。
            OSError: 目录访问失败。
            RuntimeError: 实例已关闭。
        """
        self.require_open()
        if not path or "\x00" in path:
            raise FileToolError("invalid_argument", "Provide a nonempty file path.")
        relative = Path(path)
        if ".." in relative.parts:
            raise FileToolError("permission_denied", "Use a path inside the Session workdir.")
        if relative.is_absolute():
            try:
                relative = relative.relative_to(self.root)
            except ValueError:
                raise FileToolError(
                    "permission_denied", "Use a path inside the Session workdir."
                ) from None
        if not relative.parts:
            raise FileToolError("not_file", "Provide a regular text file path.")
        with _file_lock(self.root, relative):
            directory_fd = os.dup(self._root_fd)
            try:
                for part in relative.parts[:-1]:
                    if create:
                        try:
                            os.mkdir(part, mode=0o700, dir_fd=directory_fd)
                        except FileExistsError:
                            pass
                    child_fd = os.open(
                        part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory_fd
                    )
                    os.close(directory_fd)
                    directory_fd = child_fd
                yield directory_fd, relative.name, str(relative)
            except OSError as exc:
                if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                    raise FileToolError(
                        "permission_denied", "Symlinks and non-directory parents are not allowed."
                    ) from None
                raise
            finally:
                os.close(directory_fd)

    @staticmethod
    def read_bytes(directory_fd: int, name: str) -> tuple[bytes, os.stat_result] | None:
        """读取有界的普通文件，缺失时返回 None。

        Args:
            directory_fd: 已授权且打开的父目录描述符。
            name: 不包含目录分隔符的文件名。

        Returns:
            原始字节与元信息；文件不存在时返回 None。

        Raises:
            FileToolError: 文件过大、不是普通文件或读取时发生变化。
            OSError: 文件打开或读取失败。
        """
        try:
            file_fd = os.open(
                name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd
            )
        except FileNotFoundError:
            return None
        with os.fdopen(file_fd, "rb") as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise FileToolError("not_file", "Use a regular text file.")
            if info.st_size > MAX_FILE_BYTES:
                raise FileToolError("file_too_large", "Use a format-specific or streaming tool.")
            content = source.read(MAX_FILE_BYTES + 1)
            if len(content) > MAX_FILE_BYTES:
                raise FileToolError("file_too_large", "Use a format-specific or streaming tool.")
            if _version(info) != _version(os.fstat(source.fileno())):
                raise FileToolError("file_changed", "Read the target again before retrying.")
        return content, info

    def replace_bytes(
        self,
        directory_fd: int,
        name: str,
        content: bytes,
        expected: tuple[bytes, os.stat_result] | None,
    ) -> None:
        """在同目录完整写入临时文件，复查版本后原子发布。

        Args:
            directory_fd: 已授权且锁定的父目录描述符。
            name: 目标文件名。
            content: 要写入的完整字节。
            expected: 操作开始时读取的内容与元信息；None 表示只能新建。

        Raises:
            FileToolError: 文件过大或目标版本发生变化。
            OSError: 权限、写入或发布失败；临时文件会清理。
        """
        if len(content) > MAX_FILE_BYTES:
            raise FileToolError("file_too_large", "Keep text files within the 16 MiB limit.")
        if expected is not None and not expected[1].st_mode & 0o222:
            raise PermissionError(errno.EACCES, "Target file is read-only")
        temporary = f".file-tool-{uuid4().hex}.tmp"
        file_fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        try:
            with os.fdopen(file_fd, "wb") as output:
                output.write(content)
                output.flush()
                if expected is not None:
                    os.fchmod(output.fileno(), stat.S_IMODE(expected[1].st_mode) & 0o777)
                os.fsync(output.fileno())
            current = self.read_bytes(directory_fd, name)
            if (current is None) != (expected is None) or (
                current is not None
                and expected is not None
                and (current[0] != expected[0] or _version(current[1]) != _version(expected[1]))
            ):
                raise FileToolError("file_changed", "Read the target again before retrying.")
            if expected is None:
                # link 的目标必须不存在，防止创建过程覆盖外部刚刚写入的文件。
                os.link(
                    temporary,
                    name,
                    src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            else:
                os.replace(temporary, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        finally:
            try:
                os.unlink(temporary, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
