"""验证批量 cell 工具的原子写入、执行顺序、部分失败和取消。"""

import asyncio
from pathlib import Path

import nbformat
import pytest
from test_notebook_tools import execute_source, make_session, output_text

from factor_demo.autoresearch_agent_v2.python_tools import PythonTools
from factor_demo.autoresearch_agent_v2.tools import NotebookTools
from factor_demo.autoresearch_agent_v2.tools.registry import ToolRegistry


async def write_cells(session, sources, **options):
    """通过公开工具批量写入测试源码。"""
    return await session.call("notebook__write_cells", {
        "notebook_name": "batch", "cells": [{"source": source} for source in sources], **options,
    })


async def run_cells(session, cells, **options):
    """通过公开工具执行写入结果中的编号和版本。"""
    return await session.call("notebook__run_cells", {
        "notebook_name": "batch", "cells": cells, **options,
    })


def test_batch_insert_is_ordered_and_returned_cells_execute_directly(tmp_path):
    """整批插在锚点之后，不执行写入内容，返回值可直接用于执行。"""
    async def scenario():
        async with make_session(tmp_path) as session:
            created = await session.call("notebook__create_notebook", {"notebook_name": "batch"})
            initial = await write_cells(session, ["x = 100", "x = 999"])
            inserted = await write_cells(
                session, ["x = 1", "x += 2", "x"],
                after_cell_id=initial["cells"][0]["cell_id"],
            )
            assert inserted["count"] == 3
            assert len({cell["cell_id"] for cell in inserted["cells"]}) == 3
            document = nbformat.read(created["path"], as_version=4)
            assert [cell.source for cell in document.cells] == [
                "x = 100", "x = 1", "x += 2", "x", "x = 999",
            ]
            assert all(cell.execution_count is None and not cell.outputs for cell in document.cells)
            result = await run_cells(session, inserted["cells"])
            assert result["status"] == "completed" and result["executed_count"] == 3
            assert output_text(result["results"][-1]) == "3"
            assert [item["execution_count"] for item in result["results"]] == [1, 2, 3]
            saved = nbformat.read(created["path"], as_version=4)
            assert saved.cells[3].outputs[0].data["text/plain"] == "3"
            # 参数顺序可以与文档顺序不同，不能由工具自行排序。
            reordered = await run_cells(session, [initial["cells"][1], inserted["cells"][-1]])
            assert output_text(reordered["results"][-1]) == "999"

    asyncio.run(scenario())


@pytest.mark.parametrize("cells", [
    [], [{"source": 1}], [{"source": "x = 1"}, {"source": None}],
    [{"source": "", "unknown": True}], [{"source": ""}] * 101,
])
def test_invalid_batch_write_is_rejected_before_creating_files(tmp_path, cells):
    """整批请求在进入写入流程前校验，空批次、类型和未知字段均拒绝。"""
    async def scenario():
        async with make_session(tmp_path) as session:
            result = await session.call("notebook__write_cells", {
                "notebook_name": "batch", "cells": cells,
            })
            assert result["error"] == "invalid_argument"
            assert list(tmp_path.iterdir()) == []

    asyncio.run(scenario())


def test_write_failure_leaves_document_unchanged(tmp_path, monkeypatch):
    """无效锚点、批次后部参数错误及保存失败都不留下前半批内容。"""
    async def scenario():
        async with make_session(tmp_path) as session:
            created = await session.call("notebook__create_notebook", {"notebook_name": "batch"})
            path = Path(created["path"])
            before = path.read_bytes()
            missing = await write_cells(session, ["x = 1", "x = 2"], after_cell_id="missing")
            assert missing["error"] == "cell_not_found"
            invalid = await session.call("notebook__write_cells", {
                "notebook_name": "batch", "cells": [{"source": "x = 1"}, {"source": 5}],
            })
            assert invalid["error"] == "invalid_argument"
            tools = session.resource(NotebookTools)

            def fail_save(*args):
                """模拟原子保存开始前的磁盘错误。"""
                raise OSError("test disk error")

            monkeypatch.setattr(tools, "_save_document", fail_save)
            failure = await write_cells(session, ["x = 1", "x = 2"])
            assert failure["error"] == "notebook_io_error"
            assert path.read_bytes() == before

    asyncio.run(scenario())


def test_batch_preflight_rejects_late_conflict_without_executing_first_cell(tmp_path):
    """后部版本或编号错误不能先执行前面的 cell。"""
    async def scenario():
        async with make_session(tmp_path) as session:
            await session.call("notebook__create_notebook", {"notebook_name": "batch"})
            written = await write_cells(session, [
                "from pathlib import Path\nPath('side-effect.txt').write_text('changed')", "2",
            ])
            requests = written["cells"]
            conflict = await run_cells(session, [requests[0], {**requests[1], "revision": 2}])
            assert conflict["error"] == "revision_conflict"
            missing = await run_cells(session, [requests[0], {"cell_id": "missing", "revision": 1}])
            assert missing["error"] == "cell_not_found"
            invalid = await run_cells(session, [{**requests[0], "revision": "1"}])
            assert invalid["error"] == "invalid_argument"
            for timeout in (0, -1, float("nan"), float("inf"), 121):
                assert (await run_cells(session, requests, timeout_s=timeout))["error"] == "invalid_argument"
            assert not (tmp_path / "side-effect.txt").exists()
            catalog = await session.call("notebook__list_cells", {"notebook_name": "batch"})
            assert all(cell["status"] == "never_executed" for cell in catalog["cells"])

    asyncio.run(scenario())


def test_python_failure_stops_batch_and_preserves_results_and_partial_memory(tmp_path):
    """错误前的结果落盘，错误后的项明确跳过，Python 副作用不回滚。"""
    async def scenario():
        async with make_session(tmp_path) as session:
            created = await session.call("notebook__create_notebook", {"notebook_name": "batch"})
            written = await write_cells(session, ["x = 10", "x += 1\nraise ValueError('stop')", "x = 999"])
            result = await run_cells(session, written["cells"])
            assert result["status"] == "error" and result["connected"] is True
            assert result["executed_count"] == 2
            assert [item["status"] for item in result["results"]] == ["completed", "error", "skipped"]
            assert result["results"][2]["reason"] == "error"
            saved = nbformat.read(created["path"], as_version=4)
            assert saved.cells[0].metadata.agent.status == "completed"
            assert saved.cells[1].outputs[-1].ename == "ValueError"
            assert saved.cells[2].metadata.agent.get("status") is None
            assert output_text(await execute_source(session, "batch", "x")) == "11"

    asyncio.run(scenario())


def test_timeout_is_shared_and_stops_remaining_cells(tmp_path, monkeypatch):
    """执行预算递减，超时清理 kernel，剩余项不执行。"""
    async def scenario():
        async with make_session(tmp_path) as session:
            created = await session.call("notebook__create_notebook", {"notebook_name": "batch"})
            written = await write_cells(session, ["x = 42", "while True: pass", "x = 999"])
            kernel = session.resource(NotebookTools)._kernels["batch"]
            execute = kernel.execute
            budgets = []

            async def observed(source, timeout_s):
                """保留真实 kernel 执行，只记录每项实际分配的剩余预算。"""
                budgets.append(timeout_s)
                return await execute(source, timeout_s)

            monkeypatch.setattr(kernel, "execute", observed)
            result = await run_cells(session, written["cells"], timeout_s=0.6)
            assert result["status"] == "timeout" and result["connected"] is False
            assert [item["status"] for item in result["results"]] == ["completed", "timeout", "skipped"]
            assert 0 < budgets[1] < budgets[0] <= 0.6
            saved = nbformat.read(created["path"], as_version=4)
            assert saved.cells[1].metadata.agent.status == "timeout"
            assert saved.cells[2].execution_count is None

    asyncio.run(scenario())


def test_cancellation_keeps_completed_and_current_cell_states(tmp_path):
    """取消批次会传播取消，保存当前状态，并保留已完成结果。"""
    async def scenario():
        async with make_session(tmp_path) as session:
            created = await session.call("notebook__create_notebook", {"notebook_name": "batch"})
            written = await write_cells(session, [
                "x = 42", "from pathlib import Path\nPath('running.txt').touch()\nwhile True: pass", "x = 999",
            ])
            task = asyncio.create_task(run_cells(session, written["cells"]))
            try:
                async with asyncio.timeout(5):
                    while not (tmp_path / "running.txt").exists():
                        await asyncio.sleep(0.01)
            finally:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            saved = nbformat.read(created["path"], as_version=4)
            assert saved.cells[0].metadata.agent.status == "completed"
            assert saved.cells[1].metadata.agent.status == "cancelled"
            assert saved.cells[2].execution_count is None
            assert not session.resource(NotebookTools)._kernels["batch"].connected

    asyncio.run(scenario())


def test_batch_tools_are_inherited_by_python_tools_and_respect_allowlist(tmp_path):
    """PythonTools 自动包含批量工具，授权列表可单独排除批量执行。"""
    async def scenario():
        session = make_session(tmp_path)
        registry = ToolRegistry()
        registry.register_stateful(PythonTools)
        session = type(session)(
            session.loop.model, session.state, tool_registry=registry,
            allow_tools=["finish", "notebook__write_cells"],
        )
        async with session:
            names = {schema["function"]["name"] for schema in session.schemas()}
            assert names == {"finish", "notebook__write_cells"}
            assert "notebook__run_cells" in PythonTools.tool_names
            with pytest.raises(PermissionError):
                await run_cells(session, [{"cell_id": "cell", "revision": 1}])

    asyncio.run(scenario())


def test_batch_save_failure_reports_prior_results_and_execution_uncertainty(tmp_path, monkeypatch):
    """执行后保存失败时保留此前结果，并说明当前项可能已有副作用。"""
    async def scenario():
        async with make_session(tmp_path) as session:
            created = await session.call("notebook__create_notebook", {"notebook_name": "batch"})
            written = await write_cells(session, ["x = 1", "x = 2", "x = 3"])
            tools = session.resource(NotebookTools)
            save = tools._save_document
            saves = 0

            def fail_second_save(*args):
                """仅让第二项执行后的保存失败，不影响第一项落盘。"""
                nonlocal saves
                saves += 1
                if saves == 2:
                    raise OSError("test save failure")
                return save(*args)

            with monkeypatch.context() as patch:
                patch.setattr(tools, "_save_document", fail_second_save)
                result = await run_cells(session, written["cells"])
            assert result["status"] == "error" and result["executed_count"] == 1
            assert [item["status"] for item in result["results"]] == ["completed", "error", "skipped"]
            assert result["results"][1]["execution_uncertain"] is True
            saved = nbformat.read(created["path"], as_version=4)
            assert saved.cells[0].metadata.agent.status == "completed"
            assert saved.cells[1].execution_count is None
            assert output_text(await execute_source(session, "batch", "x")) == "2"

    asyncio.run(scenario())
