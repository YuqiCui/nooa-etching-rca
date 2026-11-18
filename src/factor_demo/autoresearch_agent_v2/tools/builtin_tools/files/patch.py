"""将 dsh/Codex 补丁解析和文本匹配移植为无文件副作用的 Python 实现。

来源及修改说明见同目录 PATCH_LICENSE；保留上游的 MIT 与 Apache-2.0 授权。
Copyright 2025 OpenAI. Copyright (c) 2026 DeepSeek.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ....utils.file import FileToolError

_BEGIN = "*** Begin Patch"
_END = "*** End Patch"
_EOF = "*** End of File"
_HEADERS = {"*** Add File: ": "add", "*** Delete File: ": "delete", "*** Update File: ": "update"}
_SPACE = "\t\n\v\f\r \x85\xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000"
_PUNCTUATION = str.maketrans(
    {
        **dict.fromkeys("‐‑‒–—―−", "-"),
        **dict.fromkeys("‘’‚‛", "'"),
        **dict.fromkeys("“”„‟", '"'),
        **dict.fromkeys(
            "\xa0\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000", " "
        ),
    }
)


@dataclass
class PatchHunk:
    """保存原始补丁行号、上下文锚点和替换行。"""

    line: int
    old: list[str] = field(default_factory=list)
    new: list[str] = field(default_factory=list)
    context: str | None = None
    eof: bool = False


@dataclass
class PatchOperation:
    """保存按输入顺序排列的文件操作。"""

    kind: str
    path: str
    line: int
    content: str = ""
    destination: str | None = None
    hunks: list[PatchHunk] = field(default_factory=list)


def _syntax(line: int, hint: str) -> None:
    """返回不包含文件正文的语法错误。"""
    raise FileToolError("invalid_patch", hint, patch_line=line)


def _ensure_update(operation: PatchOperation | None, line: int) -> None:
    """拒绝没有修改区域或最后区域为空的更新操作。"""
    if operation is None or operation.kind != "update":
        return
    if not operation.hunks:
        _syntax(operation.line, "An update needs a nonempty hunk.")
    if not (operation.hunks[-1].old or operation.hunks[-1].new):
        _syntax(line, "An update needs a nonempty hunk.")


def _update_line(operation: PatchOperation, raw: str, number: int) -> None:
    """按 dsh 的批量解析规则累计更新区域。"""
    text = raw.rstrip(_SPACE)
    previous = operation.hunks[-1] if operation.hunks else None
    marker = text == "@@" or text.startswith("@@ ")
    if previous is not None and previous.eof:
        if not text:
            return
        if not marker:
            _syntax(number, "Use @@ after an End of File marker.")
    if previous is None and operation.destination is None and text.startswith("*** Move to: "):
        operation.destination = text.removeprefix("*** Move to: ")
        return
    if marker:
        if previous is not None and not (previous.old or previous.new):
            _syntax(number, "A hunk must contain context, added or removed lines.")
        operation.hunks.append(
            PatchHunk(number, context=text[3:] if text.startswith("@@ ") else None)
        )
        return
    if text == _EOF:
        if previous is not None:
            _ensure_update(operation, number)
            previous.eof = True
        return
    prefix = raw[:1]
    if raw and prefix not in {" ", "+", "-"}:
        _syntax(number, "Prefix hunk lines with a space, + or -.")
    hunk = previous or PatchHunk(number)
    if previous is None:
        operation.hunks.append(hunk)
    if prefix == "+":
        hunk.new.append(raw[1:])
    elif prefix == "-":
        hunk.old.append(raw[1:])
    else:
        hunk.old.append(raw[1:])
        hunk.new.append(raw[1:])


def parse_patch(patch: str) -> list[PatchOperation]:
    """解析 Codex 格式补丁，不访问文件。

    Args:
        patch: 包含 Begin/End Patch 的文本，可带字面 EOF 包装。

    Returns:
        按输入顺序排列的新增、删除和更新操作。

    Raises:
        FileToolError: 补丁语法错误，附带原始输入中的行号。
    """
    leading = patch[: len(patch) - len(patch.lstrip(_SPACE))]
    offset = leading.count("\n")
    lines = [
        (raw.removesuffix("\r"), offset + index + 1)
        for index, raw in enumerate(patch.strip(_SPACE).split("\n"))
    ]
    if (
        len(lines) >= 4
        and lines[0][0] in {"<<EOF", "<<'EOF'", '<<"EOF"'}
        and lines[-1][0].endswith("EOF")
    ):
        lines = lines[1:-1]
    if lines[0][0].strip(_SPACE) != _BEGIN:
        _syntax(lines[0][1], "Start with *** Begin Patch.")
    if lines[-1][0].strip(_SPACE) != _END:
        _syntax(lines[-1][1], "End with *** End Patch.")
    lines = [
        (raw.removesuffix("\r") if index < len(lines) - 1 else raw, number)
        for index, (raw, number) in enumerate(lines)
    ]
    operations: list[PatchOperation] = []
    ended = False
    for index, (raw, number) in enumerate(lines[1:], 1):
        operation = operations[-1] if operations else None
        header = (
            raw.rstrip(_SPACE)
            if operation and operation.kind == "update" and index != len(lines) - 1
            else raw.strip(_SPACE)
        )
        if ended and index != len(lines) - 1:
            if raw.strip(_SPACE):
                _syntax(number, "Unexpected content after End Patch.")
            continue
        if header == _END:
            _ensure_update(operation, number)
            ended = True
            continue
        matched = next((prefix for prefix in _HEADERS if header.startswith(prefix)), None)
        if matched:
            _ensure_update(operation, number)
            operations.append(PatchOperation(_HEADERS[matched], header[len(matched) :], number))
        elif operation and operation.kind == "add" and raw.startswith("+"):
            operation.content += raw[1:] + "\n"
        elif operation and operation.kind == "update":
            _update_line(operation, raw, number)
        else:
            _syntax(number, "Use Add File, Delete File or Update File headers.")
    return operations


def _normalize(line: str, level: int) -> str:
    """按上游顺序尝试原文、尾空白、首尾空白和标点归一化。"""
    if level == 0:
        return line
    if level == 1:
        return line.rstrip(_SPACE)
    trimmed = line.strip(_SPACE)
    return trimmed if level == 2 else trimmed.translate(_PUNCTUATION)


def _seek(lines: list[str], pattern: list[str], start: int, eof: bool) -> int | None:
    """在原文中按上游宽松匹配规则查找首个区域。"""
    if not pattern:
        return start
    last = len(lines) - len(pattern)
    for level in range(4):
        expected = [_normalize(line, level) for line in pattern]
        for index in range(last if eof else start, last + 1):
            if index >= 0 and all(
                _normalize(lines[index + j], level) == line for j, line in enumerate(expected)
            ):
                return index
    return None


def apply_hunks(before: str, hunks: list[PatchHunk]) -> str:
    """在 LF 文本上应用区域；所有搜索使用原文，非空结果以 LF 结尾。

    Args:
        before: 已由调用方统一为 LF 的文本。
        hunks: 已解析的修改区域。

    Returns:
        修改后的文本。

    Raises:
        FileToolError: 上下文或旧文本无法匹配，附带补丁行号。
    """
    original = before.split("\n")
    if original[-1] == "":
        original.pop()
    replacements: list[tuple[int, int, list[str]]] = []
    cursor = 0
    for hunk in hunks:
        if hunk.context is not None:
            found = _seek(original, [hunk.context], cursor, False)
            if found is None:
                raise FileToolError(
                    "patch_mismatch", "Context does not match; read again.", patch_line=hunk.line
                )
            cursor = found + 1
        if not hunk.old:
            start = len(original) - (1 if original and original[-1] == "" else 0)
            replacements.append((start, 0, hunk.new))
            continue
        old, new = hunk.old, hunk.new
        found = _seek(original, old, cursor, hunk.eof)
        if found is None and old[-1] == "":
            old = old[:-1]
            new = new[:-1] if new and new[-1] == "" else new
            found = _seek(original, old, cursor, hunk.eof)
        if found is None:
            raise FileToolError(
                "patch_mismatch", "Hunk does not match; read again.", patch_line=hunk.line
            )
        replacements.append((found, len(old), new))
        cursor = found + len(old)
    result = original[:]
    # 相同起点的区域须逆转补丁顺序，不能改为稳定的降序排序。
    replacements.sort(key=lambda item: item[0])
    for start, length, new in reversed(replacements):
        result[start : start + length] = new
    if not result or result[-1] != "":
        result.append("")
    return "\n".join(result)
