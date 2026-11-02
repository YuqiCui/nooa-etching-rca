"""验证 Session 的沙箱资源共享、真实工具执行和关闭后的权限撤销。"""

import asyncio
import json
import shlex
import sys
from contextlib import asynccontextmanager

import pytest
from pydantic import ValidationError

from factor_demo.autoresearch_agent_v2.agent import Session, SessionState
from factor_demo.autoresearch_agent_v2.agent.model import Model
from factor_demo.autoresearch_agent_v2.config import ModelConfig
from factor_demo.autoresearch_agent_v2.sandboxing import Sandbox, SandboxConfig
from factor_demo.autoresearch_agent_v2.tools import SandboxTools


def make_session(workdir, identity="main"):
    """构造不请求真实模型的会话。"""
    model = Model(ModelConfig(model="openai/fixture", api_key="fixture"))
    state = SessionState(session_id=identity, agent_id=identity, task_id="task", workdir=workdir)
    return Session(model, state)


def tool_call(name, arguments, identity):
    """构造用于驱动真实循环的模拟模型工具请求。"""
    return {
        "id": identity,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def test_registration_is_lazy_and_resources_are_shared_only_within_session(tmp_path):
    async def scenario():
        first, second = make_session(tmp_path), make_session(tmp_path, "child")
        first.register_stateful_tool(SandboxTools)
        SandboxTools.register(second)
        with pytest.raises(RuntimeError, match="not active"):
            first.resource(Sandbox)
        assert list(tmp_path.iterdir()) == []
        async with first, second:
            sandbox = first.resource(Sandbox)
            assert first.resource(SandboxTools).sandbox is sandbox
            assert second.resource(Sandbox) is not sandbox
            await first.start()
            assert first.resource(Sandbox) is sandbox
            names = {item["function"]["name"] for item in first.schemas()}
            assert names == {"finish", "register_dataset", "sandbox__run_shell"}
            assert list(tmp_path.iterdir()) == []
        for command in (sandbox.run_shell("echo closed"), sandbox.run_python("print('closed')")):
            with pytest.raises(RuntimeError, match="closed"):
                await command
        with pytest.raises(RuntimeError, match="not active"):
            await first.call("sandbox__run_shell", {"command": "echo closed"})

    asyncio.run(scenario())


def test_configuration_and_model_arguments_cannot_replace_session_scope(tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    session = make_session(tmp_path)
    with pytest.raises(ValueError, match="match Session"):
        SandboxTools.register(session, config=SandboxConfig(workdir=other))
    SandboxTools.register(session, config=SandboxConfig(workdir=tmp_path, timeout_s=2))

    async def scenario():
        async with session:
            assert "超时上限为 2 秒" in session.loop.build_system_prompt()
            schemas = {item["function"]["name"]: item["function"] for item in session.schemas()}
            parameters = schemas["sandbox__run_shell"]["parameters"]
            assert set(parameters["properties"]) == {"command", "timeout_s"}
            assert parameters["additionalProperties"] is False
            for arguments in (
                {"command": "echo hello", "workdir": str(other)},
                {"command": "echo hello", "backend": "process"},
                {"command": 12},
            ):
                with pytest.raises(ValidationError):
                    await session.call("sandbox__run_shell", arguments)
            with pytest.raises(ValueError, match="configured limit"):
                await session.call("sandbox__run_shell", {"command": "echo hello", "timeout_s": 3})

    asyncio.run(scenario())


def test_native_loop_returns_invalid_timeout_to_model_without_executing(tmp_path, monkeypatch):
    session = make_session(tmp_path)
    SandboxTools.register(session, config=SandboxConfig(workdir=tmp_path, timeout_s=2))
    replies = iter(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    tool_call(
                        "sandbox__run_shell", {"command": "touch forbidden", "timeout_s": 3}, "bad"
                    )
                ],
            },
            {
                "role": "assistant",
                "tool_calls": [tool_call("finish", {"content": "已修正"}, "done")],
            },
        ]
    )
    requests = []

    async def complete(messages, *, tools=None):
        requests.append(messages)
        return next(replies)

    monkeypatch.setattr(session.loop.model, "completion", complete)

    async def scenario():
        async with session:
            assert await session.run("执行") == "已修正"
            assert session.resource(Sandbox).config.timeout_s == 2

    asyncio.run(scenario())
    assert "configured limit" in requests[1][-1]["content"]
    assert requests[1][-1]["tool_call_id"] == "bad"
    assert not (tmp_path / "forbidden").exists()


@pytest.mark.skipif(sys.platform not in {"darwin", "linux"}, reason="需要真实系统沙箱")
@pytest.mark.parametrize(
    "command,timed_out", [("printf started; sleep 30", True), ("printf started; exit 7", False)]
)
def test_native_loop_returns_real_shell_failure_and_runs_next_command(
    tmp_path, monkeypatch, command, timed_out
):
    """真实 Shell 失败或超时后，模型能收到结果并继续执行下一条命令。"""
    session = make_session(tmp_path)
    SandboxTools.register(session, config=SandboxConfig(workdir=tmp_path, timeout_s=5))
    replies = iter(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    tool_call("sandbox__run_shell", {"command": command, "timeout_s": 1}, "failed")
                ],
            },
            {
                "role": "assistant",
                "tool_calls": [
                    tool_call("sandbox__run_shell", {"command": "printf recovered"}, "recovered")
                ],
            },
            {"role": "assistant", "tool_calls": [tool_call("finish", {"content": "完成"}, "done")]},
        ]
    )
    requests = []

    async def complete(messages, *, tools=None):
        requests.append(messages)
        return next(replies)

    monkeypatch.setattr(session.loop.model, "completion", complete)

    async def scenario():
        async with session:
            assert await session.run("执行") == "完成"

    asyncio.run(scenario())
    assert requests[1][-1]["tool_call_id"] == "failed"
    failure = json.loads(requests[1][-1]["content"])
    assert failure["ok"] is False
    assert failure["timed_out"] is timed_out
    assert failure["stdout"] == "started"
    if not timed_out:
        assert failure["returncode"] == 7
    assert requests[2][-1]["tool_call_id"] == "recovered"
    recovered = json.loads(requests[2][-1]["content"])
    assert recovered["ok"] is True
    assert recovered["stdout"] == "recovered"


def test_changed_workdir_fails_before_sandbox_creation(tmp_path):
    session = make_session(tmp_path)
    SandboxTools.register(session)
    other = tmp_path / "other"
    other.mkdir()
    session.state.workdir = other

    async def scenario():
        with pytest.raises(ValueError, match="match Session"):
            await session.start()
        with pytest.raises(RuntimeError, match="not active"):
            session.resource(Sandbox)
        await session.aclose()

    asyncio.run(scenario())


def test_host_provided_sandbox_must_match_session_workdir(tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    session = make_session(tmp_path)

    @asynccontextmanager
    async def wrong_scope(owner):
        sandbox = Sandbox(SandboxConfig(workdir=other))
        try:
            yield sandbox
        finally:
            await sandbox.aclose()

    session.register_resource(Sandbox, wrong_scope)
    session.register_stateful_tool(SandboxTools)

    async def scenario():
        with pytest.raises(ValueError, match="match Session"):
            await session.start()
        await session.aclose()

    asyncio.run(scenario())


def test_failed_startup_revokes_already_created_sandbox(tmp_path):
    session = make_session(tmp_path)
    SandboxTools.register(session)
    created = []

    @asynccontextmanager
    async def fail_after_sandbox(owner):
        created.append(owner.resource(Sandbox))
        raise ValueError("later resource failed")
        yield  # 创建失败的异步上下文仍需声明为生成器。

    session.register_resource(str, fail_after_sandbox)

    async def scenario():
        with pytest.raises(ValueError, match="later resource"):
            await session.start()
        with pytest.raises(RuntimeError, match="closed"):
            await created[0].run_shell("echo should-not-run")
        assert not any(
            schema["function"]["name"].startswith("sandbox__")
            for schema in session.loop.tools.schemas()
        )
        await session.aclose()

    asyncio.run(scenario())


@pytest.mark.skipif(sys.platform not in {"darwin", "linux"}, reason="需要真实系统沙箱")
def test_real_session_tools_share_files_but_cannot_read_other_session(tmp_path):
    first_root, second_root = tmp_path / "first", tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    (second_root / "private.txt").write_text("OTHER_SESSION_SECRET")

    async def scenario():
        session = make_session(first_root)
        SandboxTools.register(session)
        async with session:
            await session.call(
                "register_dataset",
                {"name": "其他会话数据", "path": str(second_root), "description": "仅登记位置"},
            )
            assert session.resource(Sandbox).config.read_paths == ()
            written = await session.call(
                "sandbox__run_shell", {"command": "printf from-shell > shared.txt"}
            )
            assert written["ok"], written["stderr"]
            read = await session.call(
                "sandbox__run_shell",
                {"command": "cat shared.txt; echo"},
            )
            assert read["ok"] and read["stdout"] == "from-shell\n"
            denied = await session.call(
                "sandbox__run_shell",
                {"command": f"cat {shlex.quote(str(second_root / 'private.txt'))}"},
            )
            assert not denied["ok"] and "OTHER_SESSION_SECRET" not in denied["stdout"]

    asyncio.run(scenario())


@pytest.mark.skipif(sys.platform not in {"darwin", "linux"}, reason="需要真实系统沙箱")
def test_real_sandbox_tools_run_in_model_loop_and_survive_followup_turn(tmp_path, monkeypatch):
    requests = []

    async def completion(self, messages, *, tools=None):
        requests.append((messages, tools))
        calls = []
        if len(requests) == 1:
            calls = [
                tool_call("sandbox__run_shell", {"command": "echo value > loop.txt"}, "shell"),
                tool_call(
                    "sandbox__run_shell",
                    {"command": "cat loop.txt"},
                    "read",
                ),
            ]
        calls.append(tool_call("finish", {"content": "done"}, f"finish-{len(requests)}"))
        return {"role": "assistant", "tool_calls": calls}

    monkeypatch.setattr(Model, "completion", completion)

    async def scenario():
        session = make_session(tmp_path)
        SandboxTools.register(session)
        try:
            assert await session.run("写入并读取文件") == "done"
            resource = session.resource(Sandbox)
            assert await session.run("继续") == "done"
            assert session.resource(Sandbox) is resource
            results = {
                message["tool_call_id"]: json.loads(message["content"])
                for message in session.state.messages
                if message["role"] == "tool" and message["tool_call_id"] in {"shell", "read"}
            }
            assert results["shell"]["ok"]
            assert results["read"]["ok"] and results["read"]["stdout"] == "value\n"
            assert "sandbox__run_shell" in {schema["function"]["name"] for schema in requests[0][1]}
        finally:
            await session.aclose()

    asyncio.run(scenario())


@pytest.mark.skipif(sys.platform not in {"darwin", "linux"}, reason="需要真实系统沙箱")
def test_session_close_cancels_borrowed_sandbox_execution(tmp_path):
    async def scenario():
        session = make_session(tmp_path)
        SandboxTools.register(session)
        await session.start()
        sandbox = session.resource(Sandbox)
        task = asyncio.create_task(
            sandbox.run_shell("echo started > started.txt; sleep 30; touch should-not-exist")
        )
        try:
            for _ in range(200):
                if (tmp_path / "started.txt").exists() or task.done():
                    break
                await asyncio.sleep(0.01)
            assert (tmp_path / "started.txt").exists(), task.result() if task.done() else "未启动"
            await asyncio.wait_for(session.aclose(), timeout=5)
            with pytest.raises(asyncio.CancelledError):
                await task
            with pytest.raises(RuntimeError, match="closed"):
                await sandbox.run_python("print('closed')")
            assert not (tmp_path / "should-not-exist").exists()
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await session.aclose()

    asyncio.run(scenario())


@pytest.mark.skipif(sys.platform not in {"darwin", "linux"}, reason="需要真实系统沙箱")
def test_shell_symlink_cannot_redirect_host_result_saving(tmp_path, monkeypatch):
    workdir, outside = tmp_path / "work", tmp_path / "outside"
    workdir.mkdir()
    outside.mkdir()

    async def completion(self, messages, *, tools=None):
        return {
            "role": "assistant",
            "tool_calls": [
                tool_call(
                    "sandbox__run_shell",
                    {
                        "command": f"ln -s {shlex.quote(str(outside))} tool_results; "
                        "printf '%30000s' x"
                    },
                    "redirect-result",
                )
            ],
        }

    monkeypatch.setattr(Model, "completion", completion)

    async def scenario():
        session = make_session(workdir)
        SandboxTools.register(session)
        async with session:
            with pytest.raises(OSError):
                await session.run("返回大结果")
        assert list(outside.iterdir()) == []

    asyncio.run(scenario())
