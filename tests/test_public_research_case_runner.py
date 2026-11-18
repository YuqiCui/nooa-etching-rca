"""核对新公开案例入口移除代码次数限制，同时保留调用审计。"""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import dotenv


def test_more_than_forty_code_calls_are_audited_and_forwarded(tmp_path, monkeypatch):
    """超过四十次仍传给原工具，不发送额外限额错误。"""
    suite = tmp_path / "suite"
    (suite / "contexts/a").mkdir(parents=True)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "argv", [
        "runner", "--suite", str(suite), "--case", "01-ai4i", "--arm", "a",
    ])
    monkeypatch.chdir(Path.cwd())
    path = Path(__file__).resolve().parents[1] / "scripts/experiments/run_public_research_case.py"
    spec = importlib.util.spec_from_file_location("public_case_runner_test", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    assert not any(name.startswith("skill__") for name in runner.ROUTER_TOOLS)
    runner.ORIGINAL_OBSERVE = lambda *args: None
    forwarded = []

    async def call(name, arguments, *, call_id=None):
        forwarded.append((name, arguments))
        return {"ok": True}

    session = SimpleNamespace(
        loop=SimpleNamespace(tools=SimpleNamespace(call=call)),
        state=SimpleNamespace(agent_id="fixture", assignment_id="assignment"),
    )
    runner.audited_tools(session, None)

    async def run():
        for _ in range(45):
            assert await session.loop.tools.call("sandbox__run_shell", {"command": "fixture"}) == {
                "ok": True,
            }

    asyncio.run(run())
    assert len(forwarded) == runner.USED == 45
    count = json.loads((runner.BASE / "execution_count.json").read_text())
    assert count == {"used": 45, "limit": None}
    rows = [json.loads(line) for line in (runner.AUDIT / "tool_events.jsonl").read_text().splitlines()]
    assert len(rows) == 90
    assert {row["event"] for row in rows} == {"start", "end"}
