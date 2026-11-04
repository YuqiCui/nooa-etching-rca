"""验证按名称管理的真实 Jupyter Notebook、会话连接和执行隔离。"""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import nbformat
import pytest

from factor_demo.autoresearch_agent_v2.agent import Session, SessionState
from factor_demo.autoresearch_agent_v2.agent.model import Model
from factor_demo.autoresearch_agent_v2.config import ModelConfig
from factor_demo.autoresearch_agent_v2.sandboxing import Sandbox, SandboxConfig
from factor_demo.autoresearch_agent_v2.sandboxing.backends import bubblewrap as bubblewrap_backend
from factor_demo.autoresearch_agent_v2.sandboxing.backends import build_command
from factor_demo.autoresearch_agent_v2.tools import NotebookTools, SandboxTools
from factor_demo.autoresearch_agent_v2.tools.builtin_tools.notebooks.kernel import NotebookKernel


def make_session(workdir, identity="main"):
    """构造不请求真实模型的 Session，使用真实 Notebook 工具。"""
    model = Model(ModelConfig(model="openai/fixture", api_key="fixture"))
    state = SessionState(session_id=identity, agent_id=identity, task_id="task", workdir=workdir)
    session = Session(model, state)
    session.register_stateful_tool(NotebookTools)
    return session


async def execute_source(session, notebook_name, source, **options):
    """通过模型实际调用的工具入口新增并执行一个 cell。"""
    cell = await session.call(
        "notebook__write_new_cell", {"notebook_name": notebook_name, "source": source}
    )
    return await session.call(
        "notebook__exec_cell",
        {"notebook_name": notebook_name, "cell_id": cell["cell_id"], "revision": 1, **options},
    )


def output_text(result):
    """提取文本和表达式结果，保留富媒体输出供其他断言检查。"""
    return "".join(
        output.get("text", "") + output.get("data", {}).get("text/plain", "")
        for output in result["outputs"]
    )


def test_schema_uses_names_and_registration_does_not_start_kernels(tmp_path):
    async def scenario():
        session = make_session(tmp_path)
        session.register_stateful_tool(SandboxTools)
        async with session:
            tools = session.resource(NotebookTools)
            assert tools.sandbox is session.resource(Sandbox)
            assert tools.sandbox is session.resource(SandboxTools).sandbox
            assert list(tmp_path.iterdir()) == []
            schemas = {
                item["function"]["name"]: item["function"]["parameters"]
                for item in session.schemas()
                if item["function"]["name"].startswith("notebook__")
            }
            assert set(schemas) == {
                "notebook__create_notebook",
                "notebook__list_notebooks",
                "notebook__write_new_cell",
                "notebook__list_cells",
                "notebook__read_cell",
                "notebook__edit_cell",
                "notebook__exec_cell",
            }
            for name, parameters in schemas.items():
                assert "notebook_id" not in parameters["properties"]
                assert parameters["additionalProperties"] is False
                if name != "notebook__list_notebooks":
                    assert "notebook_name" in parameters["required"]
            assert await session.call("notebook__list_notebooks", {}) == {
                "count": 0,
                "notebooks": [],
            }
            for name in ("../escape", "/absolute", "a/b", "a\\b", " ", "bad\nname", "x.ipynb"):
                result = await session.call("notebook__create_notebook", {"notebook_name": name})
                assert result["error"] == "invalid_argument"
        with pytest.raises(RuntimeError, match="closed"):
            await tools.list_notebooks()

    asyncio.run(scenario())


def test_multiple_notebooks_connect_on_create_and_preserve_independent_memory(tmp_path):
    async def scenario():
        async with make_session(tmp_path) as session:
            first = await session.call("notebook__create_notebook", {"notebook_name": "数据探索"})
            second = await session.call(
                "notebook__create_notebook", {"notebook_name": "comparison"}
            )
            assert first["connected"] is second["connected"] is True
            tools = session.resource(NotebookTools)
            kernels = list(tools._kernels.values())
            processes = [kernel.process for kernel in kernels]
            runtimes = [Path(kernel._temporary.name) for kernel in kernels]
            assert processes[0].pid != processes[1].pid
            for created in (first, second):
                document = nbformat.read(created["path"], as_version=4)
                nbformat.validate(document)
                assert document.cells == []
            await execute_source(session, "数据探索", "x = 40")
            assert output_text(await execute_source(session, "数据探索", "x + 2")) == "42"
            session.loop.model.completion = AsyncMock(
                return_value={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "finish-call",
                            "type": "function",
                            "function": {"name": "finish", "arguments": '{"content":"done"}'},
                        }
                    ],
                }
            )
            assert await session.run("结束本轮") == "done"
            assert await session.run("继续会话") == "done"
            assert (
                output_text(await execute_source(session, "comparison", "'x' in globals()"))
                == "False"
            )
            duplicate = await session.call(
                "notebook__create_notebook", {"notebook_name": "数据探索"}
            )
            assert duplicate["error"] == "notebook_exists"
            assert output_text(await execute_source(session, "数据探索", "x")) == "40"
            catalog = await session.call("notebook__list_notebooks", {})
            assert catalog["count"] == 2
            assert all(item["connected"] for item in catalog["notebooks"])
            assert [item["cell_count"] for item in catalog["notebooks"]] == [3, 1]
        assert all(process.returncode is not None for process in processes)
        assert all(not runtime.exists() for runtime in runtimes)
        assert Path(first["path"]).exists()

    asyncio.run(scenario())


def test_cell_editing_order_revision_checks_and_persisted_outputs(tmp_path):
    async def scenario():
        async with make_session(tmp_path) as session:
            created = await session.call("notebook__create_notebook", {"notebook_name": "analysis"})
            first = await execute_source(session, "analysis", "x = 1\nx")
            second = await session.call(
                "notebook__write_new_cell", {"notebook_name": "analysis", "source": "x"}
            )
            inserted = await session.call(
                "notebook__write_new_cell",
                {
                    "notebook_name": "analysis",
                    "source": "print('middle')",
                    "after_cell_id": first["cell_id"],
                },
            )
            cells = await session.call("notebook__list_cells", {"notebook_name": "analysis"})
            assert [cell["cell_id"] for cell in cells["cells"]] == [
                first["cell_id"],
                inserted["cell_id"],
                second["cell_id"],
            ]
            edited = await session.call(
                "notebook__edit_cell",
                {
                    "notebook_name": "analysis",
                    "cell_id": first["cell_id"],
                    "source": "x = 2\nx",
                    "expected_revision": 1,
                },
            )
            assert edited["revision"] == 2 and edited["stale"] is True
            assert edited["executed_revision"] == 1
            for tool, extra in (
                ("notebook__exec_cell", {"revision": 1}),
                ("notebook__edit_cell", {"expected_revision": 1, "source": "x = 999"}),
            ):
                failure = await session.call(
                    tool,
                    {
                        "notebook_name": "analysis",
                        "cell_id": first["cell_id"],
                        **extra,
                    },
                )
                assert failure["error"] == "revision_conflict"
            # 修改文档不会改变内存中的 x。
            assert output_text(await execute_source(session, "analysis", "x")) == "1"
            result = await session.call(
                "notebook__exec_cell",
                {
                    "notebook_name": "analysis",
                    "cell_id": first["cell_id"],
                    "revision": 2,
                },
            )
            assert result["status"] == "completed" and result["stale"] is False
            assert output_text(result) == "2"
            read = await session.call(
                "notebook__read_cell",
                {
                    "notebook_name": "analysis",
                    "cell_id": first["cell_id"],
                },
            )
            assert read["source"] == "x = 2\nx" and read["executed_revision"] == 2
            document = nbformat.read(created["path"], as_version=4)
            nbformat.validate(document)
            assert document.cells[0].outputs[0].data["text/plain"] == "2"

    asyncio.run(scenario())


def test_cell_patch_preserves_outputs_and_memory_until_execution(tmp_path):
    """局部修改只更新文档版本，执行新版本后才改变变量和输出。"""

    async def scenario():
        async with make_session(tmp_path) as session:
            created = await session.call("notebook__create_notebook", {"notebook_name": "patch"})
            first = await execute_source(session, "patch", "x = 1\nprint(x)")
            edited = await session.call(
                "notebook__edit_cell",
                {
                    "notebook_name": "patch",
                    "cell_id": first["cell_id"],
                    "expected_revision": 1,
                    "old_string": "x = 1",
                    "new_string": "x = 2",
                },
            )
            assert edited["revision"] == 2 and edited["executed_revision"] == 1
            assert edited["stale"] is True
            read = await session.call(
                "notebook__read_cell", {"notebook_name": "patch", "cell_id": first["cell_id"]}
            )
            assert read["source"] == "x = 2\nprint(x)"
            assert output_text(read) == "1\n"
            assert output_text(await execute_source(session, "patch", "x")) == "1"
            result = await session.call(
                "notebook__exec_cell",
                {
                    "notebook_name": "patch",
                    "cell_id": first["cell_id"],
                    "revision": edited["revision"],
                },
            )
            assert output_text(result) == "2\n" and result["stale"] is False
            document = nbformat.read(created["path"], as_version=4)
            assert document.cells[0].source == "x = 2\nprint(x)"
            assert document.cells[0].metadata.agent.executed_revision == 2

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("source", "patch", "expected"),
    [
        ("print('你好')\n", {"old_string": "你好", "new_string": "世界"}, "print('世界')\n"),
        ("x = 1\nprint(x)\n", {"old_string": "print(x)\n", "new_string": ""}, "x = 1\n"),
        (
            "x = 1\ny = 1\n",
            {"old_string": "= 1", "new_string": "= 2", "replace_all": True},
            "x = 2\ny = 2\n",
        ),
        ("aaa", {"old_string": "aa", "new_string": "b", "replace_all": True}, "ba"),
        ("print(1)", {"source": ""}, ""),
    ],
)
def test_cell_patch_document_edits(tmp_path, monkeypatch, source, patch, expected):
    """只模拟内核启动，验证实际文档的局部替换、删除和整段清空。"""
    monkeypatch.setattr(NotebookKernel, "start", AsyncMock())

    async def scenario():
        async with make_session(tmp_path) as session:
            await session.call("notebook__create_notebook", {"notebook_name": "patch"})
            cell = await session.call(
                "notebook__write_new_cell", {"notebook_name": "patch", "source": source}
            )
            result = await session.call(
                "notebook__edit_cell",
                {
                    "notebook_name": "patch",
                    "cell_id": cell["cell_id"],
                    "expected_revision": 1,
                    **patch,
                },
            )
            assert result["revision"] == 2 and result["stale"] is True
            read = await session.call(
                "notebook__read_cell", {"notebook_name": "patch", "cell_id": cell["cell_id"]}
            )
            assert read["source"] == expected

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("patch", "error"),
    [
        ({}, "invalid_argument"),
        ({"old_string": "x"}, "invalid_argument"),
        ({"new_string": "y"}, "invalid_argument"),
        ({"old_string": "", "new_string": "y"}, "invalid_argument"),
        ({"source": "z", "old_string": "x", "new_string": "y"}, "invalid_argument"),
        ({"source": "z", "replace_all": True}, "invalid_argument"),
        ({"old_string": "missing", "new_string": "y"}, "no_match"),
        ({"old_string": "x", "new_string": "y"}, "ambiguous_match"),
        ({"old_string": "aa", "new_string": "y"}, "ambiguous_match"),
        (
            {"old_string": "x", "new_string": "y", "expected_revision": 2},
            "revision_conflict",
        ),
    ],
)
def test_invalid_cell_patches_leave_document_unchanged(tmp_path, monkeypatch, patch, error):
    """参数、匹配或版本错误不改变文件和源码版本。"""
    monkeypatch.setattr(NotebookKernel, "start", AsyncMock())

    async def scenario():
        async with make_session(tmp_path) as session:
            created = await session.call("notebook__create_notebook", {"notebook_name": "patch"})
            cell = await session.call(
                "notebook__write_new_cell", {"notebook_name": "patch", "source": "x = x\naaa"}
            )
            before = Path(created["path"]).read_bytes()
            result = await session.call(
                "notebook__edit_cell",
                {
                    "notebook_name": "patch",
                    "cell_id": cell["cell_id"],
                    "expected_revision": 1,
                    **patch,
                },
            )
            assert result["error"] == error
            assert Path(created["path"]).read_bytes() == before

    asyncio.run(scenario())


def test_cell_patch_size_limit_and_unchanged_revision(tmp_path, monkeypatch):
    """限制补丁展开后的总长度，未改变内容时保留原版本。"""
    monkeypatch.setattr(NotebookKernel, "start", AsyncMock())

    async def scenario():
        async with make_session(tmp_path) as session:
            created = await session.call("notebook__create_notebook", {"notebook_name": "patch"})
            cell = await session.call(
                "notebook__write_new_cell", {"notebook_name": "patch", "source": "a" * 1_000_000}
            )
            arguments = {
                "notebook_name": "patch",
                "cell_id": cell["cell_id"],
                "expected_revision": 1,
            }
            before = Path(created["path"]).read_bytes()
            too_large = await session.call(
                "notebook__edit_cell",
                {**arguments, "old_string": "a", "new_string": "aa", "replace_all": True},
            )
            assert too_large["error"] == "invalid_argument"
            for replacement in (
                {"old_string": "a", "new_string": "a", "replace_all": True},
                {"source": "a" * 1_000_000},
            ):
                unchanged = await session.call("notebook__edit_cell", {**arguments, **replacement})
                assert unchanged["revision"] == 1
            assert Path(created["path"]).read_bytes() == before

    asyncio.run(scenario())


def test_python_errors_preserve_memory_and_rich_output_is_saved(tmp_path):
    async def scenario():
        async with make_session(tmp_path) as session:
            await session.call("notebook__create_notebook", {"notebook_name": "plots"})
            failed = await execute_source(session, "plots", "x = 7\nraise ValueError('test error')")
            assert failed["status"] == "error" and failed["connected"] is True
            assert failed["outputs"][-1]["ename"] == "ValueError"
            assert output_text(await execute_source(session, "plots", "x")) == "7"
            syntax = await execute_source(session, "plots", "def bad(:")
            assert syntax["status"] == "error" and syntax["connected"] is True
            image = await execute_source(
                session,
                "plots",
                (
                    "from IPython.display import display, HTML\n"
                    "display(HTML('<b>hello</b>'))\n"
                    "import matplotlib.pyplot as plt\nplt.plot([1, 2], [3, 4])\nplt.show()"
                ),
            )
            assert image["status"] == "completed"
            assert any("text/html" in output.get("data", {}) for output in image["outputs"])
            assert any("image/png" in output.get("data", {}) for output in image["outputs"])
            cleared = await execute_source(
                session,
                "plots",
                (
                    "from IPython.display import clear_output\n"
                    "print('old', flush=True)\nclear_output()\nprint('new')"
                ),
            )
            assert output_text(cleared) == "new\n"
            repeated = await execute_source(
                session,
                "plots",
                (
                    "from IPython.display import update_display\n"
                    "for _ in range(3):\n    display(HTML('small'), display_id='same')\n"
                    "update_display(HTML('x' * 100_000), display_id='same')"
                ),
            )
            assert repeated["output_truncated"] is True
            assert len(json.dumps(repeated["outputs"]).encode()) <= 262144

    asyncio.run(scenario())


def test_notebook_code_has_no_host_credentials_file_or_network_access(tmp_path, monkeypatch):
    secret = tmp_path / "secret.txt"
    secret.write_text("host secret")
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setenv("OPENAI_API_KEY", "private-fixture-key")

    async def scenario():
        async with make_session(work) as session:
            await session.call("notebook__create_notebook", {"notebook_name": "isolated"})
            result = await execute_source(
                session,
                "isolated",
                f"""
import os, socket
from pathlib import Path
assert 'OPENAI_API_KEY' not in os.environ
try:
    Path({str(secret)!r}).read_text()
except (PermissionError, FileNotFoundError):
    pass
else:
    raise AssertionError('host file exposed')
try:
    with socket.create_connection(('1.1.1.1', 443), timeout=0.5):
        pass
except OSError:
    pass
else:
    raise AssertionError('network exposed')
Path('result.txt').write_text('allowed')
print('isolated')
""",
            )
            assert result["status"] == "completed", result
            assert output_text(result) == "isolated\n"
            assert (work / "result.txt").read_text() == "allowed"

    asyncio.run(scenario())


def test_timeout_disconnects_only_affected_notebook_and_output_is_bounded(tmp_path):
    async def scenario():
        async with make_session(tmp_path) as session:
            for name in ("slow", "other"):
                await session.call("notebook__create_notebook", {"notebook_name": name})
            await execute_source(session, "other", "x = 9")
            bounded = await execute_source(session, "other", "print('x' * 400_000)")
            assert bounded["status"] == "completed" and bounded["output_truncated"] is True
            result = await execute_source(
                session,
                "slow",
                "print('before timeout', flush=True)\nwhile True: pass",
                timeout_s=0.3,
            )
            assert result["status"] == "timeout" and result["connected"] is False
            assert output_text(result) == "before timeout\n"
            assert output_text(await execute_source(session, "other", "x")) == "9"
            disconnected = await execute_source(session, "slow", "1 + 1")
            assert disconnected["error"] == "kernel_disconnected"
            items = (await session.call("notebook__list_notebooks", {}))["notebooks"]
            assert [item["connected"] for item in items] == [False, True]

    asyncio.run(scenario())


def test_cancellation_closes_kernel_and_retains_cell_record(tmp_path):
    async def scenario():
        async with make_session(tmp_path) as session:
            await session.call("notebook__create_notebook", {"notebook_name": "cancelled"})
            task = asyncio.create_task(execute_source(session, "cancelled", "while True: pass"))
            await asyncio.sleep(0.3)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            catalog = await session.call("notebook__list_notebooks", {})
            assert catalog["notebooks"][0]["connected"] is False
            cells = await session.call("notebook__list_cells", {"notebook_name": "cancelled"})
            assert cells["cells"][0]["status"] == "cancelled"

    asyncio.run(scenario())


def test_kernel_crash_is_reported_without_silently_restarting(tmp_path):
    async def scenario():
        async with make_session(tmp_path) as session:
            await session.call("notebook__create_notebook", {"notebook_name": "crash"})
            result = await execute_source(session, "crash", "import os\nos._exit(3)", timeout_s=5)
            assert result["status"] == "crashed" and result["connected"] is False

    asyncio.run(scenario())


def test_start_failure_rolls_back_and_session_names_are_separate(tmp_path, monkeypatch):
    async def scenario():
        async with make_session(tmp_path) as first, make_session(tmp_path, "other") as second:
            with monkeypatch.context() as patch:
                patch.setattr(
                    NotebookKernel, "start", AsyncMock(side_effect=RuntimeError("failed"))
                )
                failed = await first.call("notebook__create_notebook", {"notebook_name": "same"})
                assert failed["error"] == "kernel_start_failed"
                assert (await first.call("notebook__list_notebooks", {}))["count"] == 0
            one = await first.call("notebook__create_notebook", {"notebook_name": "same"})
            two = await second.call("notebook__create_notebook", {"notebook_name": "same"})
            assert one["connected"] is two["connected"] is True
            assert one["path"] != two["path"]
            await execute_source(first, "same", "x = 1")
            assert output_text(await execute_source(second, "same", "'x' in globals()")) == "False"

    asyncio.run(scenario())


def test_notebook_symlink_cannot_escape_workspace(tmp_path):
    async def scenario():
        async with make_session(tmp_path) as session:
            created = await session.call("notebook__create_notebook", {"notebook_name": "safe"})
            path = Path(created["path"])
            path.unlink()
            path.symlink_to("/etc/passwd")
            result = await session.call("notebook__list_cells", {"notebook_name": "safe"})
            assert "error" in result

    asyncio.run(scenario())


def test_linux_ipc_mounts_keep_network_namespace_isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(bubblewrap_backend.shutil, "which", lambda _: "/usr/bin/bwrap")
    ipc_dir = tmp_path / "ipc"
    ipc_dir.mkdir()
    for backend in ("bubblewrap", "landlock"):
        config = SandboxConfig(workdir=tmp_path, backend=backend)
        _, command = build_command(
            config,
            ["/usr/bin/python", "-m", "ipykernel_launcher"],
            (Path("/usr"),),
            ipc_dir=ipc_dir,
        )
        assert "--unshare-all" in command and "--share-net" not in command
        assert ["--bind", str(ipc_dir), str(ipc_dir)] == command[
            command.index(str(ipc_dir)) - 1 : command.index(str(ipc_dir)) + 2
        ]
        if backend == "landlock":
            assert json.loads(command[-1])["ipc_dir"] == str(ipc_dir)
