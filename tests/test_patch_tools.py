"""验证移植补丁的上游样例、真实文件边界、预检查和部分失败回执。"""

import asyncio
import errno
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from factor_demo.autoresearch_agent_v2.agent import Session, SessionState
from factor_demo.autoresearch_agent_v2.agent.model import Model
from factor_demo.autoresearch_agent_v2.config import ModelConfig
from factor_demo.autoresearch_agent_v2.tools import FileTools, ToolRegistry
from factor_demo.autoresearch_agent_v2.tools.builtin_tools.files.patch import (
    apply_hunks,
    parse_patch,
)
from factor_demo.autoresearch_agent_v2.utils.file import FileToolError, WorkspaceFiles

_CORPUS = json.loads((Path(__file__).parent / "fixtures/dsh_patch_cases.json").read_text())


def patch(*lines):
    """构造无需 Shell 参与的补丁文本。"""
    return "\n".join(["*** Begin Patch", *lines, "*** End Patch"])


@pytest.fixture
def files(tmp_path):
    """创建与宿主目录绑定的真实文件工具。"""
    tools = FileTools(tmp_path)
    try:
        yield tools
    finally:
        tools.files.close()


@pytest.mark.parametrize("case", _CORPUS["text"], ids=lambda case: case["name"])
def test_upstream_text_compatibility(case):
    """上游成功和失败样例使用相同匹配语义，失败不返回部分文本。"""
    operation = parse_patch(patch("*** Update File: a", *case["body"]))[0]
    if "after" in case:
        assert apply_hunks(case["before"], operation.hunks) == case["after"]
    else:
        with pytest.raises(FileToolError) as caught:
            apply_hunks(case["before"], operation.hunks)
        assert caught.value.result["error"] == "patch_mismatch"
        assert caught.value.result["patch_line"] == case["failure"]["hunkLineNumber"]


@pytest.mark.parametrize("case", _CORPUS["syntax_errors"])
def test_upstream_syntax_error_locations(case):
    """保留上游错误输入的原始行号，不在错误中回显正文。"""
    with pytest.raises(FileToolError) as caught:
        parse_patch(case["input"])
    assert caught.value.result["patch_line"] == case["line"]


def test_add_update_move_delete_in_one_patch(files, tmp_path):
    """前序新增可供后序编辑、移动和删除使用，路径与读取记录同步。"""
    result = files.apply_patch(
        patch(
            "*** Add File: nested/a.py",
            "+first",
            "+last",
            "*** Update File: nested/a.py",
            "@@",
            "-first",
            "+FIRST",
            "*** Update File: nested/a.py",
            "*** Move to: moved/a.py",
            "@@",
            "-last",
            "+LAST",
            "*** Add File: obsolete",
            "+delete me",
            "*** Delete File: obsolete",
        )
    )
    assert result["status"] == "applied"
    assert len(result["completed"]) == 5
    assert (tmp_path / "moved/a.py").read_bytes() == b"FIRST\nLAST\n"
    assert not (tmp_path / "nested/a.py").exists()
    assert not (tmp_path / "obsolete").exists()
    assert files.write_file("moved/a.py", "known")["error"] == "already_exists"
    assert files.edit_file("moved/a.py", "FIRST", "KNOWN") == "edited"
    assert "nested/a.py" not in files._reads
    assert "obsolete" not in files._reads


def test_read_state_is_not_created_by_preflight(files, tmp_path):
    """预检查读取不能授权未读文件，也不能让前序新增真正落盘。"""
    (tmp_path / "existing").write_text("old\n")
    result = files.apply_patch(
        patch(
            "*** Add File: new/first",
            "+new",
            "*** Update File: existing",
            "-old",
            "+new",
        )
    )
    assert result["error"] == "read_required"
    assert result["phase"] == "preflight"
    assert not (tmp_path / "new").exists()
    assert not files._reads
    assert (tmp_path / "existing").read_text() == "old\n"


def test_partial_read_allows_update_but_not_delete(files, tmp_path):
    """局部补丁保留未读状态；删除前必须完整读齐。"""
    (tmp_path / "a").write_text("old\ntail\n")
    files.read_file("a", 1, 1)
    assert files.apply_patch(patch("*** Update File: a", "-old", "+new"))["status"] == "applied"
    assert files.write_file("a", "bad")["error"] == "already_exists"
    assert files.apply_patch(patch("*** Delete File: a"))["error"] == "read_required"
    files.read_file("a")
    assert files.apply_patch(patch("*** Delete File: a"))["status"] == "applied"


def test_partial_update_cannot_authorize_later_delete_in_same_patch(files, tmp_path):
    """同一补丁中的局部更新不会绕过删除所需的完整读取。"""
    (tmp_path / "a").write_text("old\nunread\n")
    files.read_file("a", 1, 1)
    result = files.apply_patch(
        patch(
            "*** Update File: a",
            "-old",
            "+new",
            "*** Delete File: a",
        )
    )
    assert result["error"] == "read_required"
    assert result["phase"] == "preflight"
    assert (tmp_path / "a").read_text() == "old\nunread\n"


def test_external_change_rejects_whole_preflight(files, tmp_path):
    """补丁源版本变化时整批拒绝，不沿用宽松局部编辑策略。"""
    path = tmp_path / "a"
    path.write_text("old\n")
    files.read_file("a")
    path.write_text("old\nexternal\n")
    result = files.apply_patch(
        patch("*** Add File: new", "+new", "*** Update File: a", "-old", "+NEW")
    )
    assert result["error"] == "file_changed"
    assert not (tmp_path / "new").exists()
    assert path.read_text() == "old\nexternal\n"


@pytest.mark.parametrize(
    "body",
    [
        ["*** Update File: a", "-missing", "+new"],
        ["*** Update File: missing", "-old", "+new"],
        ["*** Add File: a", "+overwrite"],
        ["*** Update File: a", "*** Move to: occupied", "-old", "+new"],
        ["*** Add File: a/child", "+child", "*** Delete File: a"],
    ],
)
def test_preflight_refusal_leaves_all_files_unchanged(files, tmp_path, body):
    """匹配、文件存在性、移动冲突和目标层级错误不产生前序修改。"""
    (tmp_path / "a").write_text("old\n")
    (tmp_path / "occupied").write_text("keep\n")
    files.read_file("a")
    result = files.apply_patch(patch("*** Add File: should-not-exist/new", "+new", *body))
    assert "error" in result
    assert not (tmp_path / "should-not-exist").exists()
    assert (tmp_path / "a").read_text() == "old\n"
    assert (tmp_path / "occupied").read_text() == "keep\n"


@pytest.mark.parametrize("target", ["../escape", "/tmp/nooa-patch-escape", "link/file", "alias"])
def test_patch_cannot_escape_or_follow_symlinks(files, tmp_path, target):
    """新增、更新、删除和移动均不能扩大现有目录权限。"""
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.mkdir()
    (outside / "file").write_text("private\n")
    (tmp_path / "link").symlink_to(outside, target_is_directory=True)
    (tmp_path / "alias").symlink_to(outside / "file")
    for body in [
        [f"*** Add File: {target}", "+bad"],
        [f"*** Update File: {target}", "-private", "+bad"],
        [f"*** Delete File: {target}"],
        [
            "*** Add File: source",
            "+private",
            "*** Update File: source",
            f"*** Move to: {target}",
            "-private",
            "+bad",
        ],
    ]:
        result = files.apply_patch(patch(*body))
        assert result["error"] == "permission_denied"
        assert result["phase"] == "preflight"
        assert not (tmp_path / "source").exists()
    assert (outside / "file").read_text() == "private\n"


def test_move_preserves_bom_crlf_and_mode(files, tmp_path):
    """移动与编辑保留已有 BOM、统一 CRLF 和可执行位。"""
    path = tmp_path / "a"
    path.write_bytes(b"\xef\xbb\xbfold\r\ntail\r\n")
    path.chmod(0o750)
    files.read_file("a")
    result = files.apply_patch(
        patch("*** Update File: a", "*** Move to: new/b", "@@", "-old", "+new")
    )
    assert result["status"] == "applied"
    assert (tmp_path / "new/b").read_bytes() == b"\xef\xbb\xbfnew\r\ntail\r\n"
    assert (tmp_path / "new/b").stat().st_mode & 0o777 == 0o750


def test_commit_failure_keeps_completed_steps_and_skips_later_steps(files, tmp_path, monkeypatch):
    """第二步发布失败后，第一步保留，第三步不执行且回执不冒充成功。"""
    original = WorkspaceFiles.replace_bytes

    def fail_second(self, directory_fd, name, content, expected, **kwargs):
        if name == "second":
            raise OSError(errno.ENOSPC, "fixture")
        return original(self, directory_fd, name, content, expected, **kwargs)

    monkeypatch.setattr(WorkspaceFiles, "replace_bytes", fail_second)
    result = files.apply_patch(
        patch(
            "*** Add File: first",
            "+one",
            "*** Add File: second",
            "+two",
            "*** Add File: third",
            "+three",
        )
    )
    assert result["error"] == "io_error"
    assert result["phase"] == "commit"
    assert result["completed"] == [{"kind": "add", "path": "first"}]
    assert result["affected_paths"] == ["first"]
    assert result["uncertain_paths"] == []
    assert result["skipped"] == 1
    assert (tmp_path / "first").read_text() == "one\n"
    assert not (tmp_path / "second").exists()
    assert not (tmp_path / "third").exists()


def test_uncertain_publish_is_reported_and_read_state_revoked(files, tmp_path, monkeypatch):
    """文件发布后回执失败时，报告不确定路径并撤销旧读取记录。"""
    original = WorkspaceFiles.replace_bytes

    def publish_then_fail(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise OSError(errno.EIO, "fixture")

    monkeypatch.setattr(WorkspaceFiles, "replace_bytes", publish_then_fail)
    result = files.apply_patch(patch("*** Add File: a", "+new"))
    assert result["error"] == "io_error"
    assert result["uncertain_paths"] == ["a"]
    assert (tmp_path / "a").read_text() == "new\n"
    assert "a" not in files._reads


def test_partial_move_reports_created_destination(files, tmp_path, monkeypatch):
    """移动目标已发布但源删除失败时，准确列出已经创建的目标。"""
    (tmp_path / "a").write_text("old\n")
    files.read_file("a")

    def fail_delete(*args):
        raise PermissionError(errno.EACCES, "fixture")

    monkeypatch.setattr(WorkspaceFiles, "delete_bytes", fail_delete)
    result = files.apply_patch(patch("*** Update File: a", "*** Move to: b", "-old", "+new"))
    assert result["error"] == "permission_denied"
    assert result["completed"] == []
    assert result["affected_paths"] == ["b"]
    assert (tmp_path / "a").read_text() == "old\n"
    assert (tmp_path / "b").read_text() == "new\n"
    assert "a" not in files._reads and "b" not in files._reads


def test_native_session_registration_and_tool_policy(tmp_path):
    """真实会话能调用补丁，显式工具名单能隐藏补丁及其提示词。"""

    async def scenario():
        registry = ToolRegistry()
        registry.register_stateful(FileTools)
        config = ModelConfig(model="openai/fixture", api_key="fixture")
        state = SessionState(session_id="patch", agent_id="patch", task_id="t", workdir=tmp_path)
        session = Session(Model(config), state, tool_registry=registry)
        async with session:
            assert "apply_patch" in session.loop.tools.names()
            result = await session.call("apply_patch", {"input": patch("*** Add File: a", "+ok")})
            assert result["status"] == "applied"
            assert (tmp_path / "a").read_text() == "ok\n"
        with pytest.raises(RuntimeError, match="not active"):
            session.resource(FileTools)
        limited = Session(
            Model(config), state, tool_registry=registry, allow_tools=["finish", "read_file"]
        )
        async with limited:
            assert "apply_patch" not in {
                s["function"]["name"] for s in limited.loop.tools.schemas()
            }
            assert "apply_patch(input)" not in limited.loop.tools.instructions()

    asyncio.run(scenario())


@pytest.mark.parametrize("content", [b"\xff", b"binary\x00data"])
def test_binary_source_is_rejected_without_changes(files, tmp_path, content):
    """拒绝非法 UTF-8 和 NUL，预检查不触碰其他目标。"""
    (tmp_path / "a").write_bytes(content)
    result = files.apply_patch(patch("*** Add File: new", "+new", "*** Delete File: a"))
    assert result["error"] == "invalid_text"
    assert not (tmp_path / "new").exists()
    assert (tmp_path / "a").read_bytes() == content


@pytest.mark.parametrize("kind", ["update", "delete", "move"])
def test_read_only_source_rejected_in_preflight(files, tmp_path, kind):
    """只读源文件在执行任何前序新增前就被拒绝。"""
    source = tmp_path / "a"
    source.write_text("old\n")
    source.chmod(0o400)
    files.read_file("a")
    body = ["*** Delete File: a"] if kind == "delete" else ["*** Update File: a"]
    if kind == "move":
        body.append("*** Move to: b")
    if kind != "delete":
        body.extend(["-old", "+new"])
    result = files.apply_patch(patch("*** Add File: new/first", "+new", *body))
    assert result["error"] == "permission_denied"
    assert not (tmp_path / "new").exists()
    assert source.read_text() == "old\n"


def test_external_mutation_after_preflight_is_not_overwritten(files, tmp_path, monkeypatch):
    """预检查后出现的外部写入会使执行停止，保留外部内容。"""
    source = tmp_path / "a"
    source.write_text("old\n")
    files.read_file("a")
    original = WorkspaceFiles.replace_bytes

    def mutate_later_source(self, directory_fd, name, content, expected, **kwargs):
        result = original(self, directory_fd, name, content, expected, **kwargs)
        if name == "first":
            source.write_text("external\n")
        return result

    monkeypatch.setattr(WorkspaceFiles, "replace_bytes", mutate_later_source)
    result = files.apply_patch(
        patch(
            "*** Add File: first",
            "+done",
            "*** Update File: a",
            "-old",
            "+new",
        )
    )
    assert result["error"] == "file_changed"
    assert result["phase"] == "commit"
    assert result["affected_paths"] == ["first"]
    assert source.read_text() == "external\n"


def test_opposite_order_patches_share_locks_and_keep_versions(tmp_path):
    """相反顺序的跨文件补丁使用统一锁顺序，后执行者拒绝旧版本。"""
    for name in ("a", "b"):
        (tmp_path / name).write_text("old\n")
    ready = Barrier(2)

    def run(order):
        tools = FileTools(tmp_path)
        try:
            for name in order:
                tools.read_file(name)
            ready.wait(timeout=5)
            body = [line for name in order for line in (f"*** Update File: {name}", "-old", "+new")]
            return tools.apply_patch(patch(*body))
        finally:
            tools.files.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(run, [("a", "b"), ("b", "a")]))
    assert sum(result.get("status") == "applied" for result in results) == 1
    assert sum(result.get("error") == "file_changed" for result in results) == 1
    assert all((tmp_path / name).read_text() == "new\n" for name in ("a", "b"))


def test_patch_input_operation_and_memory_limits(files, tmp_path, monkeypatch):
    """输入、操作数和预检查内存预算均在写入前生效。"""
    from factor_demo.autoresearch_agent_v2.tools.builtin_tools.files import patching

    monkeypatch.setattr(patching, "_MAX_PATCH_BYTES", 10)
    assert files.apply_patch(patch())["error"] == "patch_too_large"
    monkeypatch.setattr(patching, "_MAX_PATCH_BYTES", 1024)
    monkeypatch.setattr(patching, "_MAX_OPERATIONS", 1)
    body = patch("*** Add File: a", "+one", "*** Add File: b", "+two")
    assert files.apply_patch(body)["error"] == "patch_too_large"
    monkeypatch.setattr(patching, "_MAX_OPERATIONS", 128)
    monkeypatch.setattr(patching, "_MAX_PLAN_BYTES", 1)
    assert files.apply_patch(body)["error"] == "patch_too_large"
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r\r\n"])
def test_parser_wrappers_line_endings_and_literal_shell_text(newline):
    """解析字面包装和换行，不执行路径及内容中的 Shell 文本。"""
    body = patch("*** Add File: $(literal)", "+`text` $HOME", "+second")
    operation = parse_patch("\n<<'EOF'\n" + body.replace("\n", newline) + "\nEOF\n")[0]
    assert operation.line == 4
    assert operation.path == "$(literal)"
    assert operation.content == "`text` $HOME\nsecond\n"
