"""Regressions for issue #2 items 1–4 (NPROC, cell wall-clock, seccomp, Landlock∩bwrap)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from factor_demo.research_workspace import PythonWorkspace

pytestmark = pytest.mark.skipif(
    sys.platform not in {"darwin", "linux"}, reason="enforced execution backend required"
)


def _case(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    (run / "input").mkdir(parents=True)
    (run / "workspace" / "artifacts").mkdir(parents=True)
    (run / "input" / "x.txt").write_text("data")
    return run


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux rlimits")
def test_issue2_nproc_fork_hits_eagain(tmp_path, monkeypatch):
    monkeypatch.setenv("FACTOR_CELL_NPROC", "32")
    run = _case(tmp_path)
    code = "import os, resource, time\nsoft, hard = resource.getrlimit(resource.RLIMIT_NPROC)\nprint(f'nproc={soft}:{hard}')\nkids = 0\nerr = None\nwhile kids < 200:\n    try:\n        pid = os.fork()\n    except OSError as e:\n        err = e\n        break\n    if pid == 0:\n        time.sleep(30)\n        os._exit(0)\n    kids += 1\nprint(f'spawned={kids} errno={getattr(err, \"errno\", None)}')"

    async def exercise():
        ws = PythonWorkspace(run)
        try:
            result = await ws.execute(code, 60)
            assert result["ok"], result.get("output")
            out = result["output"]
            assert "nproc=32:" in out
            assert "errno=11" in out
            spawned = int(
                [x for x in out.splitlines() if "spawned=" in x][-1].split("=")[1].split()[0]
            )
            assert spawned < 64
        finally:
            await ws.close()

    asyncio.run(exercise())


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Landlock/bwrap policy")
def test_issue2_landlock_without_pidns_fail_closed(monkeypatch):
    from factor_demo import sandboxing as sb

    monkeypatch.delenv("FACTOR_ALLOW_LANDLOCK_WITHOUT_PIDNS", raising=False)
    monkeypatch.setenv("FACTOR_LINUX_BACKEND", "landlock")
    sb.linux_backend.cache_clear()
    monkeypatch.setattr(sb, "_pid_namespace_prefix", list)
    with pytest.raises(RuntimeError, match="disabled by default"):
        sb.launch([Path("/tmp/x.py")], [], ["/tmp"])
    monkeypatch.setenv("FACTOR_ALLOW_LANDLOCK_WITHOUT_PIDNS", "1")
    cmd = sb.launch([Path("/tmp/x.py")], [], ["/tmp"])
    assert any("cell_reaper" in str(part) for part in cmd)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="seccomp")
def test_issue2_seccomp_denies_extended_syscalls(tmp_path):
    run = _case(tmp_path)
    code = "import ctypes, platform\nlibc = ctypes.CDLL(None, use_errno=True)\narch = platform.machine()\ntable = {\n 'x86_64': dict(perf_event_open=298, userfaultfd=323, bpf=321, keyctl=250,\n              add_key=248, mount=165, umount2=166, open_by_handle_at=304,\n              name_to_handle_at=303, swapon=87, reboot=169, personality=135,\n              io_uring_setup=425),\n 'aarch64': dict(perf_event_open=241, userfaultfd=282, bpf=280, keyctl=219,\n               add_key=217, mount=40, umount2=39, open_by_handle_at=265,\n               name_to_handle_at=264, swapon=224, reboot=142, personality=92,\n               io_uring_setup=425),\n}[arch]\nout = []\nfor name, nr in table.items():\n    ctypes.set_errno(0)\n    r = libc.syscall(nr, 0, 0, 0, 0, 0, 0)\n    e = ctypes.get_errno()\n    out.append(f'{name}={r}/{e}')\nprint('||'.join(out))"

    async def exercise():
        ws = PythonWorkspace(run)
        try:
            result = await ws.execute(code, 60)
            assert result["ok"], result["output"]
            line = [x for x in result["output"].splitlines() if "perf_event_open=" in x][-1]
            for name in [
                "perf_event_open",
                "userfaultfd",
                "bpf",
                "keyctl",
                "mount",
                "io_uring_setup",
            ]:
                assert f"{name}=-1/1" in line or f"{name}=-1/38" in line, line
        finally:
            await ws.close()

    asyncio.run(exercise())


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="bubblewrap+landlock")
def test_issue2_landlock_stacked_under_bubblewrap(tmp_path, monkeypatch):
    """bwrap binds /dev; Landlock only allows null/urandom/random — /dev/zero must EACCES."""
    monkeypatch.setenv("FACTOR_LINUX_BACKEND", "bubblewrap")
    from factor_demo import sandboxing as sb

    sb.linux_backend.cache_clear()
    run = _case(tmp_path)
    code = (
        "try:\n"
        "    data = open('/dev/zero', 'rb').read(1)\n"
        "    print('DEV_ZERO_OK', len(data))\n"
        "except OSError as e:\n"
        "    print('DEV_ZERO_DENIED', type(e).__name__, e.errno)\n"
    )

    async def exercise():
        ws = PythonWorkspace(run)
        try:
            result = await ws.execute(code, 60)
            assert result["ok"], result.get("output")
            out = result["output"]
            assert "DEV_ZERO_OK" not in out
            assert "DEV_ZERO_DENIED" in out
            assert "13" in out or "PermissionError" in out
        finally:
            await ws.close()

    asyncio.run(exercise())
