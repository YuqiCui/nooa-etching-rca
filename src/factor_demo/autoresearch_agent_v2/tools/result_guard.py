"""在工具结果写入消息前，将超限内容保存到文件并返回首尾预览。"""

import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

MAX_TOOL_RESULT_BYTES = 20 * 1024
PREVIEW_CHARS = 500
TRUNCATION_MARKER = "\n... [truncated] ...\n"


def _save_large_result(content: str, output_dir: Path, workspace_root: Path | None) -> Path:
    """通过目录描述符逐级拒绝符号链接，再独占创建结果文件。"""
    target = Path(os.path.abspath(output_dir))
    root = Path(os.path.abspath(workspace_root)) if workspace_root is not None else target.parent
    relative = target.relative_to(root)
    if workspace_root is None:
        root.mkdir(parents=True, exist_ok=True)
    canonical_root = root.resolve(strict=True)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_fd = os.open(canonical_root, flags)
    try:
        for part in relative.parts:
            try:
                os.mkdir(part, mode=0o700, dir_fd=directory_fd)
            except FileExistsError:
                pass
            child_fd = os.open(part, flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = child_fd
        name = f"result-{uuid4().hex}.txt"
        file_fd = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        with os.fdopen(file_fd, "w", encoding="utf-8", newline="") as output:
            output.write(content)
        return canonical_root / relative / name
    finally:
        os.close(directory_fd)


def protect_tool_result(
    result: Any, output_dir: Path, *, workspace_root: Path | None = None
) -> str:
    """生成可写入 messages 的工具结果，超限时保留完整文件。

    Args:
        result: 工具返回值；字符串直接使用，其他值先转换为 JSON 字符串。
        output_dir: 超限结果的保存目录，仅在需要落盘时创建。
        workspace_root: 可选的可信工作目录根；指定后拒绝越界路径，并逐级拒绝
            保存目录中的符号链接。None 时由调用方信任 output_dir 的父目录。

    Returns:
        不超过 20 KiB 时返回原始字符串或 JSON 字符串。超过时返回 JSON，包含
        truncated、original_size_bytes、preview 和 file_path；preview 共 500 个
        Unicode 字符，含中间截断标记，file_path 为完整 UTF-8 文本文件的绝对路径。
        文件持续保留，由工作目录的生命周期管理逻辑负责清理。

    Raises:
        TypeError: 非字符串结果无法进行 JSON 序列化。
        ValueError: 结果包含 JSON 无法序列化的循环引用。
        UnicodeError: 字符串无法编码为 UTF-8。
        OSError: 保存目录包含符号链接，或创建、写入失败；不回退到超限消息。
    """
    content = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    size_bytes = len(content.encode("utf-8"))
    if size_bytes <= MAX_TOOL_RESULT_BYTES:
        return content

    file_path = _save_large_result(content, output_dir, workspace_root)

    visible_chars = PREVIEW_CHARS - len(TRUNCATION_MARKER)
    head_chars = (visible_chars + 1) // 2
    tail_chars = visible_chars // 2
    preview = content[:head_chars] + TRUNCATION_MARKER + content[-tail_chars:]
    return json.dumps(
        {
            "truncated": True,
            "original_size_bytes": size_bytes,
            "preview": preview,
            "file_path": str(file_path),
        },
        ensure_ascii=False,
    )
