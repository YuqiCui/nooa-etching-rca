"""验证 v2 的真实 Shell 隔离、Python 包装和执行生命周期。"""

import asyncio
import json
import shlex
import socket
import sys
import time
from pathlib import Path

import psutil
import pytest

from factor_demo.autoresearch_agent_v2.sandboxing import Sandbox, SandboxConfig
from factor_demo.autoresearch_agent_v2.sandboxing.backends import bubblewrap as bubblewrap_backend
from factor_demo.autoresearch_agent_v2.sandboxing.backends import build_command
from factor_demo.autoresearch_agent_v2.sandboxing.backends.landlock import _BOOTSTRAP


@pytest.fixture(params=["native", "landlock"])
def sandbox(request, tmp_path):
    """在宿主真实后端执行；非 Linux 不运行 Landlock 集成检查。"""
    if sys.platform not in {"darwin", "linux"}:
        pytest.skip("需要 macOS 或 Linux")
    if request.param == "landlock" and sys.platform != "linux":
        pytest.skip("Landlock 集成检查需要 Linux")
    workdir = tmp_path / "work"
    workdir.mkdir()
    backend = "landlock" if request.param == "landlock" else "auto"
    instance = Sandbox(SandboxConfig(workdir=workdir, backend=backend, timeout_s=10))
    probe = asyncio.run(instance.run_shell("printf READY"))
    assert probe.ok, probe.stderr
    return instance


def test_shell_pipeline_redirection_cwd_and_exit_code(sandbox):
    result = asyncio.run(
        sandbox.run_shell("pwd; printf 'ab\\ncd\\n' | tr a-z A-Z > output.txt; cat output.txt")
    )
    assert result.ok
    assert result.stdout == f"{sandbox.config.workdir}\nAB\nCD\n"
    assert (sandbox.config.workdir / "output.txt").read_text() == "AB\nCD\n"
    failed = asyncio.run(sandbox.run_shell("printf failure >&2; exit 7"))
    assert not failed.ok and failed.returncode == 7 and failed.stderr == "failure"


def test_python_quotes_source_and_arguments_without_shell_injection(sandbox):
    argument = "a 'quoted' value; $(touch injected); `touch injected2`\nend"
    result = asyncio.run(
        sandbox.run_python(
            "import json, sys; from pathlib import Path; "
            "Path('python.txt').write_text('hello'); print(json.dumps(sys.argv[1:]))",
            args=[argument, "", "--flag"],
        )
    )
    assert result.ok, result.stderr
    assert json.loads(result.stdout) == [argument, "", "--flag"]
    assert (sandbox.config.workdir / "python.txt").read_text() == "hello"
    assert not (sandbox.config.workdir / "injected").exists()
    assert not (sandbox.config.workdir / "injected2").exists()


def test_python_scientific_dependency_is_readable(sandbox):
    result = asyncio.run(sandbox.run_python("import numpy as np; print(np.arange(4).sum())"))
    assert result.ok, result.stderr
    assert result.stdout.strip() == "6"


def test_shell_and_python_cannot_escape_workspace_or_follow_outside_symlinks(sandbox):
    outside = sandbox.config.workdir.parent / "private.txt"
    outside.write_text("PRIVATE_CONTENT")
    link = sandbox.config.workdir / "outside-link"
    link.symlink_to(outside)
    for command in (
        f"cat {shlex.quote(str(outside))}",
        "cat ../private.txt",
        "cat outside-link",
        f"printf overwritten > {shlex.quote(str(outside))}",
        "printf overwritten > outside-link",
    ):
        result = asyncio.run(sandbox.run_shell(command))
        assert not result.ok, command
        assert "PRIVATE_CONTENT" not in result.stdout
        assert outside.read_text() == "PRIVATE_CONTENT"
    denied = asyncio.run(sandbox.run_python(f"open({str(outside)!r}).read()"))
    assert not denied.ok
    assert "PRIVATE_CONTENT" not in denied.stdout


def test_authorized_input_is_readonly(sandbox):
    source = sandbox.config.workdir.parent / "input.txt"
    source.write_text("INPUT_CONTENT")
    reader = Sandbox(sandbox.config.model_copy(update={"read_paths": (source,)}))
    read = asyncio.run(reader.run_shell(f"cat {shlex.quote(str(source))}"))
    assert read.ok and read.stdout == "INPUT_CONTENT"
    write = asyncio.run(reader.run_shell(f"printf bad > {shlex.quote(str(source))}"))
    assert not write.ok
    assert source.read_text() == "INPUT_CONTENT"


def test_environment_and_inheritable_host_descriptors_are_not_exposed(sandbox, monkeypatch):
    monkeypatch.setenv("SANDBOX_TEST_SECRET", "DO_NOT_INHERIT")
    monkeypatch.setenv("BASH_ENV", "/outside/startup.sh")
    outside = sandbox.config.workdir.parent / "descriptor.txt"
    outside.write_text("HOST_FD_SECRET")
    with outside.open() as stream:
        import os

        os.set_inheritable(stream.fileno(), True)
        result = asyncio.run(
            sandbox.run_python(
                "import os; print(os.environ.get('SANDBOX_TEST_SECRET')); "
                "print(os.environ.get('BASH_ENV')); "
                f"print(os.read({stream.fileno()}, 100))"
            )
        )
    assert not result.ok
    assert result.stdout.startswith("None\nNone\n")
    assert "HOST_FD_SECRET" not in result.stdout


def test_output_limit_drains_both_streams(sandbox):
    limited = Sandbox(sandbox.config.model_copy(update={"max_output_bytes": 100}))
    result = asyncio.run(
        limited.run_python("import os; os.write(1, b'x'*1000000); os.write(2, b'y'*1000000)")
    )
    assert result.ok and result.output_truncated
    assert result.stdout == "x" * 100 and result.stderr == "y" * 100


def _assert_process_stopped(pid: int):
    """允许已退出且等待系统回收的僵尸进程，拒绝仍活跃的任务。"""
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                return
        except psutil.NoSuchProcess:
            return
        time.sleep(0.02)
    pytest.fail(f"执行结束后进程 {pid} 仍存活")


def test_timeout_kills_command_and_captures_partial_output(sandbox):
    result = asyncio.run(
        sandbox.run_shell("echo $$ > timeout.pid; printf started; sleep 30", timeout_s=1)
    )
    assert result.timed_out and not result.ok
    assert result.stdout == "started"
    assert result.duration_s < 5
    if result.backend == "macos":
        _assert_process_stopped(int((sandbox.config.workdir / "timeout.pid").read_text()))


def test_normal_exit_cleans_background_process_group(sandbox):
    result = asyncio.run(
        sandbox.run_shell("sleep 30 >/dev/null 2>&1 & echo $! > background.pid; exit 0")
    )
    assert result.ok, result.stderr
    # 私有 PID 空间的 PID 不能在宿主上直接查询，只检查 macOS 的宿主 PID。
    if result.backend == "macos":
        _assert_process_stopped(int((sandbox.config.workdir / "background.pid").read_text()))


def test_cancellation_cleans_process_and_propagates(sandbox):
    async def cancel():
        marker = sandbox.config.workdir / "cancel.pid"
        task = asyncio.create_task(sandbox.run_shell("echo $$ > cancel.pid; sleep 30"))
        for _ in range(200):
            if marker.exists():
                break
            await asyncio.sleep(0.01)
        assert marker.exists()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return int(marker.read_text())

    pid = asyncio.run(cancel())
    if sys.platform == "darwin":
        _assert_process_stopped(pid)


def test_network_is_denied_by_default_and_can_be_enabled(sandbox):
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(2)
        port = server.getsockname()[1]
        code = (
            f"import socket; socket.create_connection(('127.0.0.1', {port}), timeout=1); "
            "print('CONNECTED')"
        )
        denied = asyncio.run(sandbox.run_python(code))
        assert not denied.ok and "CONNECTED" not in denied.stdout
        connected = Sandbox(sandbox.config.model_copy(update={"allow_network": True}))
        allowed = asyncio.run(connected.run_python(code))
        assert allowed.ok and allowed.stdout.strip() == "CONNECTED", allowed.stderr


def test_python_entry_delegates_to_shell(tmp_path, monkeypatch):
    instance = Sandbox(SandboxConfig(workdir=tmp_path))
    recorded = []

    async def record(command, *, timeout_s=None):
        recorded.append((command, timeout_s))
        return "result"

    monkeypatch.setattr(instance, "run_shell", record)
    assert asyncio.run(instance.run_python("print('quoted')", args=["$(value)"], timeout_s=2)) == (
        "result"
    )
    assert shlex.split(recorded[0][0]) == [
        "exec",
        sys.executable,
        "-I",
        "-c",
        "print('quoted')",
        "$(value)",
    ]
    assert recorded[0][1] == 2


def test_linux_backends_build_native_shell_commands(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(bubblewrap_backend.shutil, "which", lambda _: "/usr/bin/bwrap")
    for name in ("bubblewrap", "landlock"):
        config = SandboxConfig(workdir=tmp_path, backend=name)
        backend, argv = build_command(config, ["/bin/sh", "-c", "echo hello"], (Path("/usr"),))
        assert backend == name and argv[0] == "/usr/bin/bwrap"
        assert "--unshare-all" in argv and "--share-net" not in argv
        assert ["--ro-bind", "/", "/"] != argv[argv.index("--ro-bind") :][:3]
        if name == "bubblewrap":
            assert argv[-3:] == ["/bin/sh", "-c", "echo hello"]
        else:
            payload = json.loads(argv[-1])
            assert payload["argv"] == ["/bin/sh", "-c", "echo hello"]
            assert payload["workdir"] == str(tmp_path.resolve())
            assert argv[-2] == _BOOTSTRAP
        network = config.model_copy(update={"allow_network": True})
        assert "--share-net" in build_command(network, ["/bin/true"], ())[1]


def test_no_unrestricted_fallback_when_linux_backend_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(bubblewrap_backend.shutil, "which", lambda _: None)
    instance = Sandbox(SandboxConfig(workdir=tmp_path))
    with pytest.raises(RuntimeError, match="bubblewrap"):
        asyncio.run(instance.run_shell("touch should-not-exist"))
    assert not (tmp_path / "should-not-exist").exists()


def test_rejects_unsafe_paths_and_timeout_escalation(tmp_path):
    with pytest.raises(ValueError):
        SandboxConfig(workdir=Path("/"))
    with pytest.raises(ValueError):
        SandboxConfig(workdir=tmp_path, read_paths=(Path("/"),))
    source = tmp_path / "readonly"
    source.write_text("data")
    with pytest.raises(ValueError, match="outside"):
        Sandbox(SandboxConfig(workdir=tmp_path, read_paths=(source,)))
    instance = Sandbox(SandboxConfig(workdir=tmp_path, timeout_s=1))
    for timeout in (0, -1, 2, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="timeout_s"):
            asyncio.run(instance.run_shell("touch should-not-exist", timeout_s=timeout))
    assert not (tmp_path / "should-not-exist").exists()
