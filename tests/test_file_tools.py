"""验证文件工具的精简协议、真实文件操作、隔离与会话接入。"""

import asyncio
import errno
import json
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from pydantic import ValidationError

from factor_demo.autoresearch_agent_v2.agent import Session, SessionState
from factor_demo.autoresearch_agent_v2.agent.model import Model
from factor_demo.autoresearch_agent_v2.config import ModelConfig
from factor_demo.autoresearch_agent_v2.tools import FileTools, ToolFactory
from factor_demo.autoresearch_agent_v2.tools.builtin_tools.files.tools import MAX_READ_BYTES
from factor_demo.autoresearch_agent_v2.tools.result_guard import protect_tool_result
from factor_demo.autoresearch_agent_v2.utils import file as file_access


@pytest.fixture
def files(tmp_path):
    """创建真实目录中的文件工具，测试后释放描述符。"""
    tools = FileTools(tmp_path)
    try:
        yield tools
    finally:
        tools.files.close()


def make_session(root, identity="s"):
    """创建不发送真实模型请求的会话，并显式注册文件工具。"""
    state = SessionState(session_id=identity, agent_id=identity, task_id="t", workdir=root)
    session = Session(Model(ModelConfig(model="openai/fixture", api_key="fixture")), state)
    session.register_stateful_tool(FileTools)
    return session


def test_create_edit_delete_and_overwrite_use_minimal_confirmations(files, tmp_path):
    """新文件不需要读取，后续精确编辑、删除和完整覆盖不回显内容。"""
    assert files.write_file("nested/a.py", "first = 1\nlast = 2") == "created"
    path = tmp_path / "nested/a.py"
    assert path.read_bytes() == b"first = 1\nlast = 2"
    assert files.edit_file("nested/a.py", "first = 1", "first = 3") == "edited"
    assert files.edit_file("nested/a.py", "last = 2", "") == "edited"
    assert path.read_bytes() == b"first = 3\n"
    assert files.write_file("nested/a.py", "replacement") == "written"
    assert path.read_bytes() == b"replacement"
    before = path.stat()
    assert files.write_file("nested/a.py", "replacement") == "unchanged"
    assert files.edit_file("nested/a.py", "replacement", "replacement") == "unchanged"
    assert path.stat().st_ino == before.st_ino
    assert path.stat().st_mtime_ns == before.st_mtime_ns


def test_read_preserves_indentation_and_paginates_from_one(files, tmp_path):
    """读取从 1 开始，行号后保留制表符、空格和末尾换行的原文。"""
    (tmp_path / "a.py").write_bytes(b"def f():\n\treturn 1\n\nlast")
    assert files.read_file("a.py", 2, 2) == "lines 2-3/4\n2\t\treturn 1\n3\t\n\nnext_offset=4"
    assert files.read_file("a.py", 4) == "lines 4-4/4\n4\tlast"
    error = files.read_file("a.py", 5)
    assert error["error"] == "offset_out_of_range"
    assert error["total_lines"] == 4
    assert files.read_file("missing")["error"] == "not_found"


def test_default_page_is_200_lines_and_ranges_must_cover_whole_version(files, tmp_path):
    """不连续或重复分页不能伪装成完整读取，读齐后才允许覆盖。"""
    path = tmp_path / "a.py"
    path.write_text("x\n" * 250, encoding="utf-8")
    assert files.write_file("a.py", "bad")["error"] == "read_required"
    result = files.read_file("a.py")
    assert result.startswith("lines 1-200/250\n")
    assert result.endswith("next_offset=201")
    files.read_file("a.py", 230)
    files.read_file("a.py", 1, 200)
    assert files.write_file("a.py", "bad")["error"] == "read_required"
    files.read_file("a.py", 201, 29)
    assert files.write_file("a.py", "ok") == "written"
    assert path.read_bytes() == b"ok"


def test_page_coverage_is_reset_when_file_changes(files, tmp_path):
    """不同版本的分页不能拼成完整读取记录。"""
    path = tmp_path / "a"
    path.write_bytes(b"a\nb\n")
    files.read_file("a", 1, 1)
    path.write_bytes(b"c\nb\n")
    files.read_file("a", 2, 1)
    assert files.write_file("a", "bad")["error"] == "read_required"
    files.read_file("a", 1, 1)
    assert files.write_file("a", "ok") == "written"


def test_empty_file_counts_as_a_complete_read(files, tmp_path):
    """空文件存在时仍需读取，空内容本身构成完整读取。"""
    (tmp_path / "a").touch()
    assert files.write_file("a", "x")["error"] == "read_required"
    assert files.read_file("a") == "empty file"
    assert files.write_file("a", "x") == "written"


def test_rendered_byte_budget_covers_multibyte_text_and_resumes_without_loss(files, tmp_path):
    """按最终 UTF-8 字节限流，连续分页重建全部原文且不会触发结果二次落盘。"""
    text = "    中文🙂\n" * 1700
    (tmp_path / "a").write_text(text, encoding="utf-8")
    offset = 1
    recovered = []
    pages = 0
    while True:
        result = files.read_file("a", offset, 10000)
        assert len(result.encode("utf-8")) <= MAX_READ_BYTES
        assert protect_tool_result(result, tmp_path / "tool_results") == result
        body = result.split("\n", 1)[1]
        if "\nnext_offset=" in body:
            body, next_offset = body.rsplit("\nnext_offset=", 1)
            offset = int(next_offset)
        else:
            next_offset = None
        recovered.extend(line.split("\t", 1)[1] for line in body.splitlines(keepends=True))
        pages += 1
        if next_offset is None:
            break
    assert pages > 1
    assert "".join(recovered) == text
    assert not (tmp_path / "tool_results").exists()
    assert files.write_file("a", "ok") == "written"


def test_long_line_is_not_truncated_or_counted_as_read(files, tmp_path):
    """超长行返回错误，不向覆盖保护提供虚假的已读记录。"""
    (tmp_path / "a").write_text("中" * MAX_READ_BYTES, encoding="utf-8")
    assert files.read_file("a")["error"] == "line_too_long"
    assert files.write_file("a", "bad")["error"] == "read_required"


@pytest.mark.parametrize("original", [b"\xff\xfe\x00x", b"a\x00b", b"\xff"])
def test_binary_and_invalid_utf8_remain_untouched(files, tmp_path, original):
    path = tmp_path / "a"
    path.write_bytes(original)
    assert files.read_file("a")["error"] == "invalid_text"
    assert files.write_file("a", "bad")["error"] == "read_required"
    assert path.read_bytes() == original


@pytest.mark.parametrize("bad_text", ["a\x00b", "\ud800"])
def test_invalid_output_text_does_not_create_a_file(files, tmp_path, bad_text):
    """拒绝 NUL 与非法 Unicode，失败不创建目标文件。"""
    assert files.write_file("a", bad_text)["error"] == "invalid_text"
    assert not (tmp_path / "a").exists()


def test_match_errors_replacement_counts_and_bounded_diagnostics(files, tmp_path):
    """匹配失败不写文件，重复匹配只返回有限行号，全部替换计数准确。"""
    path = tmp_path / "a"
    original = "x = 1\n" * 20
    path.write_text(original, encoding="utf-8")
    assert files.edit_file("a", "x", "y")["error"] == "read_required"
    files.read_file("a", 1, 1)
    assert files.edit_file("a", "", "bad")["error"] == "invalid_argument"
    assert files.edit_file("a", "missing", "bad")["error"] == "no_match"
    error = files.edit_file("a", "x = 1", "bad")
    assert error["error"] == "ambiguous_match"
    assert error["matches"] == 20
    assert error["lines"] == list(range(1, 9))
    assert path.read_text(encoding="utf-8") == original
    assert files.edit_file("a", "x = 1", "x = 2", True) == "edited 20 matches"
    assert path.read_text(encoding="utf-8") == "x = 2\n" * 20
    assert files.write_file("a", "bad")["error"] == "read_required"


def test_overlapping_matches_are_not_treated_as_unique(files):
    """存在重叠位置时默认拒绝歧义，全部替换沿用不重叠字面替换语义。"""
    files.write_file("a", "aaaa")
    error = files.edit_file("a", "aa", "b")
    assert error["error"] == "ambiguous_match"
    assert error["matches"] == 3
    assert files.edit_file("a", "aa", "b", True) == "edited 2 matches"


def test_external_changes_block_overwrite_but_allow_unambiguous_edit(files, tmp_path):
    """外部变更阻止整文件覆盖，仍唯一匹配的局部修改保留外部内容。"""
    path = tmp_path / "a"
    path.write_bytes(b"a = 1\nb = 2\n")
    files.read_file("a")
    path.write_bytes(b"a = 1\nb = 3\n")
    assert files.write_file("a", "bad")["error"] == "file_changed"
    assert files.edit_file("a", "a = 1", "a = 4") == "edited; changed_since_read=true"
    assert path.read_bytes() == b"a = 4\nb = 3\n"
    assert files.write_file("a", "bad")["error"] == "read_required"
    files.read_file("a")
    assert files.write_file("a", "ok") == "written"


def test_crlf_bom_indentation_and_executable_mode_survive_edit(files, tmp_path):
    """LF 编辑参数可修改 CRLF 文件，BOM、缩进和执行权限保持不变。"""
    path = tmp_path / "a"
    path.write_bytes(b"\xef\xbb\xbfdef f():\r\n\treturn 1\r\n")
    path.chmod(0o750)
    assert files.read_file("a") == "lines 1-2/2\n1\tdef f():\r\n2\t\treturn 1\r\n"
    assert files.edit_file("a", "def f():\n\treturn 1", "def f():\n\treturn 2") == "edited"
    assert path.read_bytes() == b"\xef\xbb\xbfdef f():\r\n\treturn 2\r\n"
    assert path.stat().st_mode & 0o777 == 0o750
    assert files.write_file("a", "new\n") == "written"
    assert path.read_bytes() == b"\xef\xbb\xbfnew\n"


@pytest.mark.parametrize("ending", [b"\n", b"\r"])
def test_mixed_newlines_require_exact_match_and_preserve_other_bytes(files, tmp_path, ending):
    """混合换行不启用 CRLF 适配，精确匹配后只改变指定片段。"""
    path = tmp_path / "a"
    original = b"first\r\nsecond" + ending + b"third"
    path.write_bytes(original)
    files.read_file("a")
    assert files.edit_file("a", "first\nsecond", "bad")["error"] == "no_match"
    assert path.read_bytes() == original
    assert files.edit_file("a", "first\r\nsecond", "updated\r\nsecond") == "edited"
    assert path.read_bytes() == b"updated\r\nsecond" + ending + b"third"


def test_relative_and_absolute_aliases_share_read_state(files, tmp_path):
    """同一文件的相对、绝对和点路径使用同一读取记录。"""
    path = tmp_path / "a"
    path.write_bytes(b"a")
    files.read_file(str(path))
    assert files.write_file("./a", "b") == "written"
    assert files.edit_file(str(path), "b", "c") == "edited"


@pytest.mark.parametrize("operation", ["read", "write", "edit"])
@pytest.mark.parametrize("kind", ["absolute", "parent", "leaf_link", "parent_link"])
def test_all_operations_reject_escape_and_symlinks(files, tmp_path, operation, kind):
    """所有入口一致拒绝路径越界，以及文件和父目录符号链接。"""
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.mkdir()
    target = outside / "secret"
    target.write_bytes(b"private")
    if kind == "absolute":
        path = str(target)
    elif kind == "parent":
        path = f"../{outside.name}/secret"
    elif kind == "leaf_link":
        (tmp_path / "link").symlink_to(target)
        path = "link"
    else:
        (tmp_path / "link").symlink_to(outside, target_is_directory=True)
        path = "link/secret"
    functions = {
        "read": lambda: files.read_file(path),
        "write": lambda: files.write_file(path, "bad"),
        "edit": lambda: files.edit_file(path, "private", "bad"),
    }
    assert functions[operation]()["error"] == "permission_denied"
    assert target.read_bytes() == b"private"
    assert sorted(p.name for p in outside.iterdir()) == ["secret"]


def test_directory_fifo_and_read_only_files_fail_without_mutation(files, tmp_path):
    """目录和 FIFO 不阻塞读取，原子替换也不能绕过只读文件保护。"""
    (tmp_path / "directory").mkdir()
    os.mkfifo(tmp_path / "fifo")
    for path in ("directory", "fifo"):
        assert files.read_file(path)["error"] == "not_file"
        assert files.write_file(path, "bad")["error"] == "not_file"
    path = tmp_path / "readonly"
    path.write_bytes(b"original")
    path.chmod(0o444)
    files.read_file("readonly")
    assert files.write_file("readonly", "bad")["error"] == "permission_denied"
    assert files.edit_file("readonly", "original", "bad")["error"] == "permission_denied"
    assert path.read_bytes() == b"original"


def test_file_size_limit_is_separate_from_output_limit(files, tmp_path, monkeypatch):
    """宿主文件大小上限独立于模型返回预算，超限不覆盖原文件。"""
    monkeypatch.setattr(file_access, "MAX_FILE_BYTES", 16)
    (tmp_path / "a").write_bytes(b"x" * 17)
    assert files.read_file("a")["error"] == "file_too_large"
    assert files.write_file("new", "x" * 17)["error"] == "file_too_large"
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("failure", ["replace", "fsync", "external_change"])
def test_failed_publish_keeps_original_and_cleans_temporary_files(
    files, tmp_path, monkeypatch, failure
):
    """发布失败或发布前发生变化时保留目标，清理所有临时文件。"""
    path = tmp_path / "a"
    path.write_bytes(b"original")
    files.read_file("a")
    original_fsync = file_access.os.fsync

    def fail_replace(*args, **kwargs):
        raise OSError(errno.ENOSPC, "simulated full disk")

    def fsync(file_fd):
        if failure == "fsync":
            raise OSError(errno.ENOSPC, "simulated full disk")
        original_fsync(file_fd)
        if failure == "external_change":
            path.write_bytes(b"external")

    monkeypatch.setattr(file_access.os, "fsync", fsync)
    if failure == "replace":
        monkeypatch.setattr(file_access.os, "replace", fail_replace)
    result = files.edit_file("a", "original", "updated")
    assert result["error"] == ("file_changed" if failure == "external_change" else "io_error")
    assert path.read_bytes() == (b"external" if failure == "external_change" else b"original")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a"]


def test_new_file_publish_does_not_overwrite_a_concurrent_creator(files, tmp_path, monkeypatch):
    """在最后一次检查后新建目标，也不能被新文件发布覆盖。"""
    original_link = file_access.os.link

    def concurrent_create(source, target, **kwargs):
        (tmp_path / "a").write_bytes(b"external")
        return original_link(source, target, **kwargs)

    monkeypatch.setattr(file_access.os, "link", concurrent_create)
    assert files.write_file("a", "bad")["error"] == "file_changed"
    assert (tmp_path / "a").read_bytes() == b"external"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a"]


def test_directory_swap_to_symlink_is_rejected(files, tmp_path, monkeypatch):
    """路径检查后父目录被换成符号链接，也不能读取外部文件。"""
    directory = tmp_path / "directory"
    directory.mkdir()
    (directory / "a").write_bytes(b"inside")
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.mkdir()
    (outside / "a").write_bytes(b"private")
    original_open = file_access.os.open

    def swap(path, flags, *args, **kwargs):
        if path == "directory" and flags & os.O_DIRECTORY:
            directory.rename(tmp_path / "moved")
            directory.symlink_to(outside, target_is_directory=True)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(file_access.os, "open", swap)
    assert files.read_file("directory/a")["error"] == "permission_denied"
    assert (outside / "a").read_bytes() == b"private"


def test_cooperating_instances_serialize_edits_across_threads(tmp_path):
    """同目录的不同工具实例通过共享锁保留彼此的独立修改。"""
    path = tmp_path / "a"
    path.write_bytes(b"left=0\nright=0\n")
    barrier = Barrier(2)

    def edit(side):
        tools = FileTools(tmp_path)
        try:
            tools.read_file("a")
            barrier.wait(timeout=5)
            return tools.edit_file("a", f"{side}=0", f"{side}=1")
        finally:
            tools.files.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(edit, ["left", "right"]))
    assert sorted(results) == ["edited", "edited; changed_since_read=true"]
    assert path.read_bytes() == b"left=1\nright=1\n"


def test_schema_validation_is_recoverable_only_when_requested(files):
    """文件参数错误不泄露输入，其他工具默认仍抛校验异常。"""
    factory = ToolFactory()
    files.register_tools(factory)
    schemas = {schema["function"]["name"]: schema["function"] for schema in factory.schemas()}
    assert set(schemas) == {"read_file", "write_file", "edit_file"}
    expected = {
        "read_file": {"path", "offset", "limit"},
        "write_file": {"path", "content"},
        "edit_file": {"path", "old_string", "new_string", "replace_all"},
    }
    for name, schema in schemas.items():
        assert set(schema["parameters"]["properties"]) == expected[name]
        assert schema["parameters"]["additionalProperties"] is False

    async def scenario():
        for arguments in (
            {},
            {"path": 3},
            {"path": "a", "offset": 0},
            {"path": "a", "limit": -1},
            {"path": "a", "offset": True},
            {"path": "a", "offset": "1"},
            {"path": "a", "extra": "private-value"},
        ):
            result = await factory.call("read_file", arguments)
            assert result["error"] == "invalid_argument"
            assert "private-value" not in json.dumps(result)
        factory.register_callable("strict_read", files.read_file, description="保留默认校验行为。")
        with pytest.raises(ValidationError):
            await factory.call("strict_read", {"path": 1})

    asyncio.run(scenario())


def test_sessions_have_independent_read_state_and_revoke_borrowed_tools(tmp_path):
    """读取记录不跨 Session 共享，关闭后已借出的实例也不能继续访问。"""
    (tmp_path / "a").write_bytes(b"x")

    async def scenario():
        first, second = make_session(tmp_path, "first"), make_session(tmp_path, "second")
        async with first, second:
            tools = first.resource(FileTools)
            await first.call("read_file", {"path": "a"})
            assert (await second.call("write_file", {"path": "a", "content": "bad"}))[
                "error"
            ] == "read_required"
            assert await first.call("write_file", {"path": "a", "content": "y"}) == "written"
        with pytest.raises(RuntimeError, match="closed"):
            tools.read_file("a")
        with pytest.raises(RuntimeError, match="not active"):
            await first.call("read_file", {"path": "a"})

    asyncio.run(scenario())


def test_agent_recovers_from_argument_and_match_errors_across_turns(monkeypatch, tmp_path):
    """模拟模型驱动真实循环，参数和匹配错误回传后仍能继续编辑并结束轮次。"""
    path = tmp_path / "a.py"
    path.write_bytes(b"value = 1\n")
    requests = []
    actions = iter(
        [
            ("read_file", {"path": "a.py", "limit": 0}),
            ("read_file", {"path": "a.py"}),
            ("edit_file", {"path": "a.py", "old_string": "missing", "new_string": "bad"}),
            ("edit_file", {"path": "a.py", "old_string": "value = 1", "new_string": "value = 2"}),
            ("finish", {"content": "done"}),
            ("write_file", {"path": "a.py", "content": "value = 3\n"}),
            ("finish", {"content": "continued"}),
        ]
    )

    async def completion(self, messages, *, tools=None):
        requests.append((messages, tools))
        name, arguments = next(actions)
        return {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": str(len(requests)),
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            ],
        }

    monkeypatch.setattr(Model, "completion", completion)

    async def scenario():
        async with make_session(tmp_path) as session:
            assert await session.run("edit") == "done"
            assert await session.run("continue") == "continued"
            history = [m["content"] for m in session.state.messages if m["role"] == "tool"]
            assert json.loads(history[0])["error"] == "invalid_argument"
            assert json.loads(history[2])["error"] == "no_match"
            assert history[3] == "edited"
            assert history[5] == "written"
            assert "diff" not in history[3]

    asyncio.run(scenario())
    assert path.read_bytes() == b"value = 3\n"
    assert {s["function"]["name"] for s in requests[0][1]} == {
        "read_file",
        "write_file",
        "edit_file",
        "finish",
        "register_dataset",
    }
