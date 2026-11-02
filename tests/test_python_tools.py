"""验证类级渐进发现、Python 调用桥接和会话执行边界。"""

import ast
import asyncio
import inspect
import json
import subprocess
import sys
import threading
from contextlib import asynccontextmanager
from copy import deepcopy

import nbformat
import psutil
import pytest

from factor_demo.autoresearch_agent_v2.agent import Session, SessionState
from factor_demo.autoresearch_agent_v2.agent.model import Model
from factor_demo.autoresearch_agent_v2.config import ModelConfig
from factor_demo.autoresearch_agent_v2.python_tools import PythonTool, PythonTools
from factor_demo.autoresearch_agent_v2.python_tools.builtin_tools.statistics import Statistics
from factor_demo.autoresearch_agent_v2.python_tools.rendering import render_class_documentation
from factor_demo.autoresearch_agent_v2.python_tools.schema import ToolMethod
from factor_demo.autoresearch_agent_v2.python_tools.typing import ClassDetails, MethodDetails
from factor_demo.autoresearch_agent_v2.sandboxing import Sandbox, SandboxConfig


def make_session(workdir, identity="main"):
    """创建使用占位模型的会话，不请求真实模型。"""
    state = SessionState(session_id=identity, agent_id=identity, task_id="task", workdir=workdir)
    session = Session(Model(ModelConfig(model="openai/fixture", api_key="fixture")), state)
    session.register_stateful_tool(PythonTools)
    return session


async def execute_source(session, options, notebook_name="analysis"):
    """通过真实 Notebook 工具写入并执行代码，保留原始执行结果。"""
    catalog = await session.call("notebook__list_notebooks", {})
    if notebook_name not in [item["notebook_name"] for item in catalog["notebooks"]]:
        created = await session.call("notebook__create_notebook", {"notebook_name": notebook_name})
        assert created["connected"], created
    cell = await session.call(
        "notebook__write_new_cell", {"notebook_name": notebook_name, "source": options["code"]}
    )
    return await session.call(
        "notebook__exec_cell",
        {
            "notebook_name": notebook_name,
            "cell_id": cell["cell_id"],
            "revision": cell["revision"],
            **{key: value for key, value in options.items() if key != "code"},
        },
    )


def output_text(result):
    """提取 Notebook 文本输出与表达式结果。"""
    return "".join(
        output.get("text", "") + output.get("data", {}).get("text/plain", "")
        for output in result["outputs"]
    )


def error_text(result):
    """提取 Notebook 中的代码异常，供断言检查。"""
    return "\n".join(
        output.get("ename", "") + ": " + output.get("evalue", "")
        for output in result["outputs"]
        if output["output_type"] == "error"
    )


def tool_call(name, arguments, identity):
    """构造驱动真实循环的模拟模型请求。"""
    return {
        "id": identity,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


@pytest.fixture
def counter_tool(monkeypatch):
    """登记带可观察生命周期的工具，测试结束后恢复全局目录。"""
    monkeypatch.setattr(PythonTool, "_registry", PythonTool.registered())
    instances, events = [], []

    @PythonTool.register("Counter")
    class Counter(PythonTool):
        """在当前会话保存计数，并演示同步和异步工具。

        方法：
            add：增加计数。
            read：读取计数。
            fail：演示工具异常。
            invalid_result：演示返回值校验。
            wait：等待异步操作完成。
        """

        def __init__(self):
            """初始化属于本实例的状态。"""
            self.value = 0
            self.started = asyncio.Event()
            self.cancelled = False

        @classmethod
        @asynccontextmanager
        async def bind(cls, session):
            """延迟创建工具，记录正常关闭和取消后的资源释放。"""
            instance = cls()
            instances.append(instance)
            events.append("opened")
            try:
                yield instance
            finally:
                events.append("closed")

        def methods(self):
            """仅公开显式列出的方法，生命周期方法不暴露。"""
            return {
                "add": self.add,
                "read": self.read,
                "fail": self.fail,
                "invalid_result": self.invalid_result,
                "wait": self.wait,
            }

        def add(self, hidden_parameter: int, scale: int = 1) -> int:
            """增加当前会话的计数。

            Args:
                hidden_parameter: 完整接口中才出现的增量，DETAIL_ONLY_PARAMETER。
                scale: 增量倍数，默认一倍。

            Returns:
                增加后的计数。DETAIL_END_MARKER。
            """
            self.value += hidden_parameter * scale
            return self.value

        def read(self) -> int:
            """读取当前计数。

            Returns:
                当前会话实例保存的计数。
            """
            return self.value

        def fail(self) -> int:
            """演示执行异常。

            Raises:
                ValueError: 固定抛出用于回归测试的错误。
            """
            raise ValueError("可处理的业务错误")

        def invalid_result(self) -> int:
            """演示错误的返回类型。

            Returns:
                声明为整数，测试实现故意违反声明。
            """
            return "bad"

        async def wait(self) -> int:
            """等待完成，取消时记录宿主异步方法已经退出。

            Returns:
                等待结束后的当前计数。
            """
            self.started.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                self.cancelled = True
                raise
            return self.value

    return Counter, instances, events


def test_builtin_catalog_loads_on_import_in_a_fresh_process():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from factor_demo.autoresearch_agent_v2.python_tools import PythonTool; "
                "assert 'Statistics' in PythonTool.registered()"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_catalog_is_class_docstrings_only_and_instances_are_lazy(tmp_path, counter_tool):
    counter, instances, events = counter_tool

    async def scenario():
        async with make_session(tmp_path) as session:
            assert not instances
            instructions = session.loop.tools.instructions()
            assert inspect.getdoc(counter) in instructions
            assert "hidden_parameter" not in instructions
            assert "DETAIL_ONLY_PARAMETER" not in instructions
            assert "Args:" not in instructions
            assert "input_schema" not in instructions
            schemas = session.schemas()
            assert {item["function"]["name"] for item in schemas} == {
                "finish",
                "register_dataset",
                "python__describe_class",
                "notebook__create_notebook",
                "notebook__list_notebooks",
                "notebook__write_new_cell",
                "notebook__list_cells",
                "notebook__read_cell",
                "notebook__edit_cell",
                "notebook__exec_cell",
            }
            with pytest.raises(KeyError):
                await session.call("python__execute", {"code": "print(1)"})
            assert "hidden_parameter" not in json.dumps(schemas)
            first = await session.call("python__describe_class", {"class_name": "Counter"})
            second = await session.call("python__describe_class", {"class_name": "Counter"})
            assert first == second
            assert len(instances) == 1
            assert first.startswith("```python\nclass Counter:\n")
            assert first.endswith("\n```")
            parsed = ast.parse(first.removeprefix("```python\n").removesuffix("\n```"))
            rendered_class = parsed.body[0]
            assert ast.get_docstring(rendered_class) == inspect.getdoc(counter)
            methods = {
                method.name: method
                for method in rendered_class.body
                if isinstance(method, ast.FunctionDef)
            }
            assert set(methods) == {"add", "read", "fail", "invalid_result", "wait"}
            assert ast.get_docstring(methods["add"]) == inspect.getdoc(counter.add)
            assert "DETAIL_END_MARKER" in first
            assert "def add(hidden_parameter: int, scale: int = 1) -> int:" in first
            assert "input_schema" not in first
            assert "output_schema" not in first
            assert "\\n" not in first
            assert "hidden_parameter" not in session.loop.tools.instructions()
        assert events == ["opened", "closed"]

    asyncio.run(scenario())


def test_real_loop_inserts_catalog_then_full_details_into_history(
    tmp_path, counter_tool, monkeypatch
):
    requests = []

    async def completion(self, messages, *, tools=None):
        requests.append((deepcopy(messages), deepcopy(tools)))
        step = len(requests)
        if step == 1:
            call = tool_call("python__describe_class", {"class_name": "Counter"}, "describe")
        elif step == 2:
            call = tool_call("notebook__create_notebook", {"notebook_name": "analysis"}, "create")
        elif step == 3:
            call = tool_call(
                "notebook__write_new_cell",
                {
                    "notebook_name": "analysis",
                    "source": "saved = Counter.add(3, scale=2)\nprint(saved)",
                },
                "write",
            )
        elif step == 4:
            cell = json.loads(messages[-1]["content"])
            call = tool_call(
                "notebook__exec_cell",
                {
                    "notebook_name": "analysis",
                    "cell_id": cell["cell_id"],
                    "revision": cell["revision"],
                },
                "run",
            )
        else:
            call = tool_call("finish", {"content": "done"}, "finish")
        return {"role": "assistant", "tool_calls": [call]}

    monkeypatch.setattr(Model, "completion", completion)

    async def scenario():
        async with make_session(tmp_path) as session:
            assert await session.run("增加计数") == "done"
            assert "hidden_parameter" not in json.dumps(requests[0], ensure_ascii=False)
            assert "DETAIL_ONLY_PARAMETER" in json.dumps(requests[1], ensure_ascii=False)
            description = next(
                message for message in requests[1][0] if message.get("tool_call_id") == "describe"
            )
            assert description["content"].startswith("```python\nclass Counter:\n")
            assert "\n        Args:\n" in description["content"]
            assert "input_schema" not in description["content"]
            result = next(
                message
                for message in session.state.messages
                if message.get("tool_call_id") == "run"
            )
            assert output_text(json.loads(result["content"])) == "6\n"
            notebook = (await session.call("notebook__list_notebooks", {}))["notebooks"][0]
            document = nbformat.read(notebook["path"], as_version=4)
            assert len(document.cells) == 1
            assert document.cells[0].source == "saved = Counter.add(3, scale=2)\nprint(saved)"
            assert document.cells[0].execution_count == 1
            assert output_text(await execute_source(session, {"code": "saved"})) == "6"
            assert "Counter" in requests[0][0][0]["content"]

    asyncio.run(scenario())


def test_loading_is_per_session_and_variables_are_per_notebook(tmp_path, counter_tool):
    async def scenario():
        async with make_session(tmp_path) as first, make_session(tmp_path, "other") as second:
            denied = await execute_source(first, {"code": "Counter.add(1)"})
            assert denied["status"] != "completed" and "NameError" in error_text(denied)
            await first.call("python__describe_class", {"class_name": "Counter"})
            executed = await execute_source(first, {"code": "saved = Counter.add(4); print(saved)"})
            assert executed["status"] == "completed" and output_text(executed) == "4\n"
            again = await execute_source(first, {"code": "print(Counter.read())"})
            assert again["status"] == "completed" and output_text(again) == "4\n"
            assert output_text(await execute_source(first, {"code": "saved"})) == "4"
            fresh = await execute_source(first, {"code": "saved"}, notebook_name="fresh")
            assert fresh["status"] == "error" and "NameError" in error_text(fresh)
            shared_tool = await execute_source(
                first, {"code": "Counter.read()"}, notebook_name="fresh"
            )
            assert output_text(shared_tool) == "4"
            await execute_source(first, {"code": "install = 123\nnamespace = 456"})
            preserved = await execute_source(first, {"code": "(install, namespace, saved)"})
            assert output_text(preserved) == "(123, 456, 4)"
            captured = await execute_source(
                first, {"code": "%%capture captured\nprint(Counter.read())"}
            )
            assert captured["status"] == "completed"
            assert output_text(await execute_source(first, {"code": "captured.stdout"})) == "'4\\n'"
            for session, code in (
                (first, "Statistics.summarize([1])"),
                (second, "Counter.read()"),
            ):
                result = await execute_source(session, {"code": code})
                assert result["status"] != "completed" and "NameError" in error_text(result)
            await second.call("python__describe_class", {"class_name": "Counter"})
            independent = await execute_source(second, {"code": "print(Counter.read())"})
            assert output_text(independent) == "0\n"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "code, error",
    [
        ("Counter.add('3')", "ValidationError"),
        ("Counter.add(3, extra=True)", "ValidationError"),
        ("Counter.add()", "ValidationError"),
        ("Counter.add(3, hidden_parameter=4)", "ValidationError"),
        ("Counter.invalid_result()", "ValidationError"),
        ("Counter.fail()", "可处理的业务错误"),
    ],
)
def test_argument_return_validation_and_errors_reach_python(tmp_path, counter_tool, code, error):
    async def scenario():
        async with make_session(tmp_path) as session:
            await session.call("python__describe_class", {"class_name": "Counter"})
            result = await execute_source(session, {"code": code})
            assert result["status"] != "completed" and error in error_text(result)
            unchanged = await execute_source(session, {"code": "print(Counter.read())"})
            assert output_text(unchanged) == "0\n"
            recovered = await execute_source(
                session,
                {"code": "try:\n Counter.fail()\nexcept RuntimeError:\n print(Counter.add(2))"},
            )
            assert recovered["status"] == "completed" and output_text(recovered) == "2\n"

    asyncio.run(scenario())


@pytest.mark.parametrize("class_name, method", [("Counter", "read"), ("Statistics", "methods")])
def test_raw_protocol_cannot_bypass_loaded_class_and_public_method_checks(
    tmp_path, counter_tool, class_name, method
):
    payload = {
        "class_name": class_name,
        "method_name": method,
        "args": [],
        "kwargs": {},
    }
    code = (
        "import json\n"
        f"response = input('\\x1ePYTHON_TOOL:' + {json.dumps(payload)!r})\n"
        "print(json.loads(response)['error']['type'])"
    )

    async def scenario():
        async with make_session(tmp_path) as session:
            await session.call("python__describe_class", {"class_name": "Statistics"})
            result = await execute_source(session, {"code": code})
            assert result["status"] == "completed" and output_text(result) == "PermissionError\n"
            assert not counter_tool[1]

    asyncio.run(scenario())


def test_threaded_python_calls_keep_each_response_with_its_request(tmp_path):
    async def scenario():
        async with make_session(tmp_path) as session:
            await session.call("python__describe_class", {"class_name": "Statistics"})
            result = await execute_source(
                session,
                {
                    "code": "from concurrent.futures import ThreadPoolExecutor\n"
                    "with ThreadPoolExecutor(4) as pool:\n"
                    " result = list(pool.map(lambda x: Statistics.summarize([x])['mean'], range(20)))\n"
                    "print(result)"
                },
            )
            assert result["status"] == "completed", error_text(result)
            assert json.loads(output_text(result)) == list(range(20))

    asyncio.run(scenario())


def test_runtime_preserves_sandbox_scientific_imports_files_and_credentials(tmp_path, monkeypatch):
    workdir = tmp_path / "work"
    workdir.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("PRIVATE_CONTENT")
    monkeypatch.setenv("PYTHON_TOOLS_TEST_SECRET", "PRIVATE_ENV")

    async def scenario():
        async with make_session(workdir) as session:
            await session.call("python__describe_class", {"class_name": "Statistics"})
            result = await execute_source(
                session,
                {
                    "code": "import numpy as np, os\n"
                    "from pathlib import Path\n"
                    "Path('result.txt').write_text('saved')\n"
                    "print(Statistics.summarize(np.arange(4).tolist())['mean'])\n"
                    "print(os.environ.get('PYTHON_TOOLS_TEST_SECRET'))\n"
                    "print('notebook-output')"
                },
            )
            assert result["status"] == "completed", error_text(result)
            assert output_text(result) == "1.5\nNone\nnotebook-output\n"
            assert (workdir / "result.txt").read_text() == "saved"
            for code in (
                f"print(open({str(secret)!r}).read())",
                "import socket; socket.create_connection(('127.0.0.1', 9), timeout=1)",
            ):
                denied = await execute_source(session, {"code": code})
                assert denied["status"] != "completed"
                assert "PRIVATE_CONTENT" not in output_text(denied)

    asyncio.run(scenario())


def test_output_budget_applies_to_prints_and_stderr(tmp_path):
    async def scenario():
        session = make_session(tmp_path)

        @asynccontextmanager
        async def limited(owner):
            sandbox = Sandbox(SandboxConfig(workdir=tmp_path, max_output_bytes=100))
            try:
                yield sandbox
            finally:
                await sandbox.aclose()

        session.register_resource(Sandbox, limited)
        async with session:
            result = await execute_source(
                session, {"code": "import sys; print('x'*100000); sys.stderr.write('y'*100000)"}
            )
            assert result["status"] == "completed" and result["output_truncated"]
            assert len(json.dumps(result["outputs"]).encode("utf-8")) <= 100

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel", [False, True])
def test_timeout_and_cancellation_stop_async_tool_and_execution_process(
    tmp_path, counter_tool, cancel
):
    async def scenario():
        async with make_session(tmp_path) as session:
            await session.call("python__describe_class", {"class_name": "Counter"})
            await session.call("notebook__create_notebook", {"notebook_name": "analysis"})
            instance = counter_tool[1][0]
            running = asyncio.create_task(
                execute_source(
                    session,
                    {
                        "code": "import os; open('worker.pid', 'w').write(str(os.getpid())); Counter.wait()",
                        "timeout_s": 0.5,
                    },
                )
            )
            await asyncio.wait_for(instance.started.wait(), timeout=5)
            if cancel:
                running.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await running
            else:
                result = await running
                assert result["status"] == "timeout" and result["status"] != "completed"
            assert instance.cancelled
            disconnected = await execute_source(session, {"code": "1 + 1"})
            assert disconnected["error"] == "kernel_disconnected"
            cells = await session.call("notebook__list_cells", {"notebook_name": "analysis"})
            assert cells["cells"][0]["status"] == ("cancelled" if cancel else "timeout")
            pid = int((tmp_path / "worker.pid").read_text())
            assert not psutil.pid_exists(pid)
            after = await execute_source(
                session, {"code": "print(Counter.read())"}, notebook_name="recovery"
            )
            assert after["status"] == "completed" and output_text(after) == "0\n"
        assert counter_tool[2] == ["opened", "closed"]

    asyncio.run(asyncio.wait_for(scenario(), timeout=30))


def test_failed_details_roll_back_instance_and_never_unlock_class(tmp_path, counter_tool):
    counter, instances, events = counter_tool

    @PythonTool.register("Broken")
    class Broken(counter):
        """演示公开方法文档不合格时的加载回滚。"""

        def methods(self):
            """将无类型声明的方法放入公开清单。"""
            return {"broken": lambda value: value}

    async def scenario():
        async with make_session(tmp_path) as session:
            with pytest.raises(ValueError, match="完整文档"):
                await session.call("python__describe_class", {"class_name": "Broken"})
            assert len(instances) == 1 and events == ["opened", "closed"]
            result = await execute_source(session, {"code": "Broken.broken(1)"})
            assert result["status"] != "completed" and "NameError" in error_text(result)
            with pytest.raises(ValueError, match="未知"):
                await session.call("python__describe_class", {"class_name": "Missing"})

    asyncio.run(scenario())


def test_closed_session_rejects_old_python_tool_reference(tmp_path, counter_tool):
    async def scenario():
        async with make_session(tmp_path) as session:
            runtime = session.resource(PythonTools)
            await session.call("python__describe_class", {"class_name": "Counter"})
        for operation in (runtime.describe_class("Statistics"), runtime.list_notebooks()):
            with pytest.raises(RuntimeError, match="关闭|closed"):
                await operation

    asyncio.run(scenario())


def test_registry_rejects_duplicates_invalid_names_and_abstract_classes(counter_tool):
    counter = counter_tool[0]
    with pytest.raises(ValueError, match="已经注册"):
        PythonTool.register("Counter")(counter)
    for name in ("", "bad-name", "_private", "class", "print"):
        with pytest.raises(ValueError, match="无效"):
            PythonTool.register(name)(counter)
    with pytest.raises(TypeError, match="具体子类"):
        PythonTool.register("Abstract")(PythonTool)


def test_builtin_methods_use_plain_types_and_validate_actual_values():
    async def scenario():
        statistics = Statistics()
        summary = ToolMethod("summarize", statistics.summarize)
        correlation = ToolMethod("pearson_correlation", statistics.pearson_correlation)
        assert "list[float]" in correlation.details.signature
        assert "dict[str, int | float]" in correlation.details.signature
        assert "$defs" not in correlation.details.input_schema
        assert "$defs" not in correlation.details.output_schema
        assert await summary.call([[2, 4, 6]], {}) == {
            "count": 3,
            "mean": 4.0,
            "median": 4.0,
            "minimum": 2.0,
            "maximum": 6.0,
            "std": pytest.approx(1.6329931619),
        }
        actual = await correlation.call([[1, 2, 3, 4], [2, 4, 6, 8]], {})
        assert actual["count"] == 4 and actual["coefficient"] == pytest.approx(1)
        for args in (([1, 2], [2, 4]), ([1, 2, 3], [1, 2, 3, 4]), ([1, 1, 1], [1, 2, 3])):
            with pytest.raises(ValueError):
                await correlation.call(list(args), {})
        for values in ([], [float("nan")], [float("inf")]):
            with pytest.raises(ValueError):
                await summary.call([values], {})

    asyncio.run(scenario())


def test_python_documentation_preserves_quotes_backslashes_and_complete_docstrings():
    class_doc = '描述路径 C:\\tools 和三引号 """。\n\n保留第二段。'
    method_doc = '读取数值。\n\nArgs:\n    value: 输入 """ 和路径 C:\\new。\n\nReturns:\n    原值。'
    details = ClassDetails(
        class_name="Example",
        docstring=class_doc,
        methods=[
            MethodDetails(
                name="read",
                signature="read(value: str = 'default') -> str",
                docstring=method_doc,
                input_schema={"type": "object"},
                output_schema={"type": "string"},
            )
        ],
    )
    rendered = render_class_documentation(details)
    parsed = ast.parse(rendered.removeprefix("```python\n").removesuffix("\n```"))
    assert ast.get_docstring(parsed.body[0]) == class_doc
    assert ast.get_docstring(parsed.body[0].body[1]) == method_doc
    assert "def read(value: str = 'default') -> str:" in rendered
    assert "input_schema" not in rendered and "output_schema" not in rendered


def test_cancelling_sync_tool_waits_for_resource_use_to_end():
    started, release = threading.Event(), threading.Event()

    def slow() -> int:
        """模拟不能强制中断的同步工具。

        Returns:
            同步工作完成后的结果。
        """
        started.set()
        release.wait(timeout=5)
        return 1

    async def scenario():
        running = asyncio.create_task(ToolMethod("slow", slow).call([], {}))
        assert await asyncio.to_thread(started.wait, 2)
        running.cancel()
        await asyncio.sleep(0.02)
        assert not running.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await running

    asyncio.run(scenario())
